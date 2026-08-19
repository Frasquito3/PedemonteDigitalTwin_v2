from __future__ import annotations

from math import isfinite
from pathlib import Path
from typing import Any, Protocol, Sequence

from planner.core.schema import (
    InstanciaTurno,
    PedidoInput,
    PlanTurno,
    Turno,
)
from planner.domain.validator import (
    validar_instancia,
    validar_plan,
)
from planner.integration.alpyne_codec import (
    PROTOCOL_VERSION,
    codificar_plan_alpyne,
)


CABECERA = 8
CAMPOS_PEDIDO = 10
MAX_PEDIDOS = 30


class PlanificadorCompatible(
    Protocol
):
    ultima_decision: Any

    def generar_plan(
        self,
        instancia: InstanciaTurno,
    ) -> PlanTurno:
        ...


_planificador: (
    PlanificadorCompatible
    | None
) = None

_modelo: Path | None = None

_ultima_decision = (
    "SIN_DECISION"
)


def inicializar(
    model_path: str,
    max_pedidos: int = 30,
    deterministic: bool = True,
) -> str:
    """
    Carga el modelo RL una sola vez dentro del proceso
    persistente de Pypeline.
    """
    global _planificador
    global _modelo
    global _ultima_decision

    ruta = (
        Path(
            model_path
        )
        .expanduser()
        .resolve()
    )

    if not ruta.is_file():
        raise FileNotFoundError(
            "No existe el modelo RL: "
            f"{ruta}"
        )

    if max_pedidos <= 0:
        raise ValueError(
            "max_pedidos debe ser > 0."
        )

    if (
        _planificador is not None
        and _modelo == ruta
    ):
        return (
            "OK|REUTILIZADO|"
            f"modelo={ruta}"
        )

    # Imports diferidos:
    # las pruebas del protocolo no necesitan
    # cargar Stable-Baselines3.
    from planner.algorithms.hybrid_rl_greedy import (
        HybridRLGreedyPlanner,
    )
    from planner.rl.planner import (
        RLPlanner,
    )

    planner_rl = RLPlanner(
        model_path=ruta,
        max_pedidos=max_pedidos,
        deterministic=deterministic,
    )

    _planificador = (
        HybridRLGreedyPlanner(
            planner_rl=planner_rl
        )
    )

    _modelo = ruta

    _ultima_decision = (
        "SIN_DECISION"
    )

    return (
        "OK|CARGADO|"
        f"modelo={ruta}"
    )


def reiniciar() -> str:
    global _planificador
    global _modelo
    global _ultima_decision

    _planificador = None

    _modelo = None

    _ultima_decision = (
        "SIN_DECISION"
    )

    return "OK|REINICIADO"


def obtener_estado() -> str:
    if _planificador is None:
        return "NO_INICIALIZADO"

    return (
        "INICIALIZADO|"
        f"modelo={_modelo}"
    )


def obtener_ultima_decision() -> str:
    return _ultima_decision


def planificar_vector(
    instancia_vector: Sequence[float],
    seed_escenario: int,
    seed_ejecucion: int,
) -> list[float]:
    """
    Reconstruye la instancia, ejecuta el híbrido
    RL–Greedy y devuelve el plan usando el protocolo
    vectorial ya validado en la Fase 10C.
    """
    global _ultima_decision

    if _planificador is None:
        raise RuntimeError(
            "El planificador Pypeline "
            "no fue inicializado."
        )

    instancia = (
        decodificar_instancia_vector(
            instancia_vector,
            seed_escenario,
            seed_ejecucion,
        )
    )

    plan = (
        _planificador
        .generar_plan(
            instancia
        )
    )

    validacion = validar_plan(
        instancia,
        plan,
    )

    if not validacion.valido:
        raise RuntimeError(
            "El híbrido produjo un plan inválido: "
            + " | ".join(
                validacion.errores
            )
        )

    _ultima_decision = (
        _decision_como_texto(
            _planificador
            .ultima_decision
        )
    )

    return codificar_plan_alpyne(
        instancia,
        plan,
    )


def decodificar_instancia_vector(
    datos: Sequence[float],
    seed_escenario: int,
    seed_ejecucion: int,
) -> InstanciaTurno:
    valores = [
        float(
            valor
        )
        for valor in datos
    ]

    if len(
        valores
    ) < CABECERA:
        raise ValueError(
            "instanciaVector no contiene "
            "la cabecera completa."
        )

    version = _entero(
        valores[0],
        "version",
    )

    if (
        version
        != PROTOCOL_VERSION
    ):
        raise ValueError(
            "Versión no soportada: "
            f"{version}."
        )

    turno_codigo = _entero(
        valores[1],
        "turno",
    )

    if turno_codigo == 0:
        turno = Turno.MANANA

        inicio = 450

        fin = 720

    elif turno_codigo == 1:
        turno = Turno.TARDE

        inicio = 840

        fin = 1020

    else:
        raise ValueError(
            "turno debe valer 0 o 1."
        )

    cantidad = _entero(
        valores[2],
        "cantidadPedidos",
    )

    capacidad = _entero(
        valores[3],
        "capacidadCamion",
    )

    camiones = _entero(
        valores[4],
        "cantidadCamiones",
    )

    if not (
        1
        <= cantidad
        <= MAX_PEDIDOS
    ):
        raise ValueError(
            "Cantidad de pedidos "
            "fuera de rango: "
            f"{cantidad}."
        )

    if (
        capacidad <= 0
        or camiones <= 0
    ):
        raise ValueError(
            "Capacidad y cantidad de "
            "camiones deben ser > 0."
        )

    longitud_esperada = (
        CABECERA
        +
        cantidad
        * CAMPOS_PEDIDO
    )

    if (
        len(
            valores
        )
        != longitud_esperada
    ):
        raise ValueError(
            "Longitud esperada="
            f"{longitud_esperada}, "
            "recibida="
            f"{len(valores)}."
        )

    lat_corralon = _rango(
        valores[5],
        -90.0,
        90.0,
        "latCorralon",
    )

    lon_corralon = _rango(
        valores[6],
        -180.0,
        180.0,
        "lonCorralon",
    )

    pedidos: list[
        PedidoInput
    ] = []

    for indice in range(
        cantidad
    ):
        base = (
            CABECERA
            +
            indice
            * CAMPOS_PEDIDO
        )

        pedido_indice = _entero(
            valores[base],
            (
                f"pedido[{indice}]"
                ".indice"
            ),
        )

        original = _entero(
            valores[base + 1],
            (
                f"pedido[{indice}]"
                ".original"
            ),
        )

        numero_parte = _entero(
            valores[base + 2],
            "numeroParte",
        )

        total_partes = _entero(
            valores[base + 3],
            "totalPartes",
        )

        unidades = _entero(
            valores[base + 4],
            "unidades",
        )

        volcador = _booleano(
            valores[base + 5],
            "requiereVolcador",
        )

        latitud = _rango(
            valores[base + 6],
            -90.0,
            90.0,
            "latitud",
        )

        longitud = _rango(
            valores[base + 7],
            -180.0,
            180.0,
            "longitud",
        )

        desde_raw = _entero(
            valores[base + 8],
            "horaDesde",
        )

        hasta_raw = _entero(
            valores[base + 9],
            "horaHasta",
        )

        if (
            pedido_indice
            != indice
        ):
            raise ValueError(
                "Los índices de pedido "
                "deben ser consecutivos."
            )

        if not (
            0
            <= original
            < cantidad
        ):
            raise ValueError(
                "Índice de pedido original "
                "fuera de rango."
            )

        if not (
            total_partes > 0
            and
            1
            <= numero_parte
            <= total_partes
        ):
            raise ValueError(
                "Numeración de partes "
                "inválida."
            )

        if not (
            1
            <= unidades
            <= capacidad
        ):
            raise ValueError(
                "Unidades fuera de la "
                "capacidad del camión."
            )

        if (
            desde_raw == -1
        ) != (
            hasta_raw == -1
        ):
            raise ValueError(
                "Una ventana ausente debe "
                "usar -1 en ambos límites."
            )

        tiene_ventana = (
            desde_raw != -1
        )

        desde = (
            desde_raw
            if tiene_ventana
            else inicio
        )

        hasta = (
            hasta_raw
            if tiene_ventana
            else fin
        )

        if not (
            inicio
            <= desde
            < hasta
            <= fin
        ):
            raise ValueError(
                "Ventana horaria "
                "fuera del turno."
            )

        pedidos.append(
            PedidoInput(
                pedido_id=(
                    f"PY-P"
                    f"{indice + 1:03d}"
                ),
                pedido_original_id=(
                    f"PY-O"
                    f"{original + 1:03d}"
                ),
                numero_parte=(
                    numero_parte
                ),
                total_partes=(
                    total_partes
                ),
                turno=turno,
                latitud=latitud,
                longitud=longitud,
                unidades_capacidad=(
                    unidades
                ),
                requiere_volcador=(
                    volcador
                ),
                tiene_ventana_especifica=(
                    tiene_ventana
                ),
                hora_desde_min=desde,
                hora_hasta_min=hasta,
                cliente=(
                    "Cliente Pypeline "
                    f"{indice + 1}"
                ),
                direccion=(
                    "Dirección Pypeline "
                    f"{indice + 1}"
                ),
                barrio="PYPELINE",
            )
        )

    instancia = InstanciaTurno(
        instancia_id=(
            "PYPELINE-"
            f"{seed_escenario}-"
            f"{seed_ejecucion}"
        ),
        fecha_operacion=(
            "1970-01-01"
        ),
        turno=turno,
        pedidos=pedidos,
        lat_corralon=(
            lat_corralon
        ),
        lon_corralon=(
            lon_corralon
        ),
        capacidad_camion=(
            capacidad
        ),
        cantidad_camiones=(
            camiones
        ),
        hora_inicio_turno_min=(
            inicio
        ),
        hora_fin_objetivo_min=(
            fin
        ),
        hora_fin_tolerancia_min=(
            fin + 15
        ),
        seed_escenario=int(
            seed_escenario
        ),
        seed_ejecucion=int(
            seed_ejecucion
        ),
    )

    errores = validar_instancia(
        instancia
    )

    if errores:
        raise ValueError(
            "Instancia inválida: "
            + " | ".join(
                errores
            )
        )

    return instancia


def _decision_como_texto(
    decision: Any,
) -> str:
    if decision is None:
        return "SIN_DECISION"

    fuente = getattr(
        getattr(
            decision,
            "fuente_seleccionada",
            None,
        ),
        "value",
        "?",
    )

    motivo = getattr(
        getattr(
            decision,
            "motivo",
            None,
        ),
        "value",
        "?",
    )

    return (
        f"fuente={fuente}|"
        f"motivo={motivo}|"
        "costo_rl="
        f"{getattr(decision, 'costo_rl', None)}|"
        "costo_greedy="
        f"{getattr(decision, 'costo_greedy', None)}|"
        "tiempo_total_ms="
        f"{getattr(decision, 'tiempo_total_ms', None)}"
    )


def _entero(
    valor: float,
    campo: str,
) -> int:
    numero = float(
        valor
    )

    if (
        not isfinite(
            numero
        )
        or abs(
            numero
            - round(
                numero
            )
        ) > 1e-9
    ):
        raise ValueError(
            f"{campo} debe ser "
            "un entero finito."
        )

    return int(
        round(
            numero
        )
    )


def _booleano(
    valor: float,
    campo: str,
) -> bool:
    entero = _entero(
        valor,
        campo,
    )

    if entero not in (
        0,
        1,
    ):
        raise ValueError(
            f"{campo} debe valer 0 o 1."
        )

    return entero == 1


def _rango(
    valor: float,
    minimo: float,
    maximo: float,
    campo: str,
) -> float:
    numero = float(
        valor
    )

    if (
        not isfinite(
            numero
        )
        or not (
            minimo
            <= numero
            <= maximo
        )
    ):
        raise ValueError(
            f"{campo} está fuera de rango."
        )

    return numero