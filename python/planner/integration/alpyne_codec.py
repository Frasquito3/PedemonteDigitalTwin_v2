from __future__ import annotations

from math import isfinite

from planner.core.schema import (
    AlgoritmoPlanificacion,
    InstanciaTurno,
    PlanTurno,
    Turno,
)

from planner.domain.validator import (
    validar_instancia,
    validar_plan,
)


PROTOCOL_VERSION = 1

TAMANO_CABECERA_INSTANCIA = 8
TAMANO_PEDIDO_INSTANCIA = 10

TAMANO_CABECERA_PLAN = 5
TAMANO_ASIGNACION_PLAN = 4

MAX_PEDIDOS_ALPYNE = 30


class ErrorContratoAlpyne(
    ValueError
):
    """Error de validación previo al envío hacia AnyLogic."""


CODIGO_ALGORITMO: dict[
    AlgoritmoPlanificacion,
    int,
] = {
    AlgoritmoPlanificacion.RL: 0,
    AlgoritmoPlanificacion.GA: 1,
    AlgoritmoPlanificacion.GREEDY: 2,
    AlgoritmoPlanificacion.RANDOM: 3,
    AlgoritmoPlanificacion.MANUAL_TEST: 4,
}


def codificar_instancia_alpyne(
    instancia: InstanciaTurno,
) -> list[float]:
    """
    Convierte una InstanciaTurno al vector double[] del
    protocolo Alpyne v1 utilizado por AnyLogic.
    """
    errores = validar_instancia(
        instancia
    )

    if errores:
        raise ErrorContratoAlpyne(
            "La instancia es inválida: "
            + " | ".join(
                errores
            )
        )

    cantidad_pedidos = len(
        instancia.pedidos
    )

    if cantidad_pedidos <= 0:
        raise ErrorContratoAlpyne(
            "La instancia debe contener al menos "
            "un pedido."
        )

    if (
        cantidad_pedidos
        > MAX_PEDIDOS_ALPYNE
    ):
        raise ErrorContratoAlpyne(
            "La instancia supera el máximo del "
            f"contrato Alpyne: {cantidad_pedidos} "
            f"> {MAX_PEDIDOS_ALPYNE}."
        )

    _exigir_entero_positivo(
        "capacidad_camion",
        instancia.capacidad_camion,
    )

    _exigir_entero_positivo(
        "cantidad_camiones",
        instancia.cantidad_camiones,
    )

    _exigir_latitud(
        "lat_corralon",
        instancia.lat_corralon,
    )

    _exigir_longitud(
        "lon_corralon",
        instancia.lon_corralon,
    )

    turno_codigo = (
        0
        if instancia.turno
        == Turno.MANANA
        else 1
    )

    indices_originales: dict[
        str,
        int,
    ] = {}

    vector: list[float] = [
        float(
            PROTOCOL_VERSION
        ),
        float(
            turno_codigo
        ),
        float(
            cantidad_pedidos
        ),
        float(
            instancia.capacidad_camion
        ),
        float(
            instancia.cantidad_camiones
        ),
        float(
            instancia.lat_corralon
        ),
        float(
            instancia.lon_corralon
        ),
        0.0,
    ]

    for indice, pedido in enumerate(
        instancia.pedidos
    ):
        original_id = (
            pedido.pedido_original_id.strip()
            or pedido.pedido_id
        )

        if (
            original_id
            not in indices_originales
        ):
            indices_originales[
                original_id
            ] = len(
                indices_originales
            )

        original_indice = (
            indices_originales[
                original_id
            ]
        )

        _exigir_entero_positivo(
            (
                f"pedido[{indice}]"
                ".numero_parte"
            ),
            pedido.numero_parte,
        )

        _exigir_entero_positivo(
            (
                f"pedido[{indice}]"
                ".total_partes"
            ),
            pedido.total_partes,
        )

        _exigir_entero_positivo(
            (
                f"pedido[{indice}]"
                ".unidades_capacidad"
            ),
            pedido.unidades_capacidad,
        )

        if (
            pedido.unidades_capacidad
            > instancia.capacidad_camion
        ):
            raise ErrorContratoAlpyne(
                f"Pedido {pedido.pedido_id}: "
                "supera la capacidad del camión."
            )

        _exigir_latitud(
            (
                f"pedido[{indice}]"
                ".latitud"
            ),
            pedido.latitud,
        )

        _exigir_longitud(
            (
                f"pedido[{indice}]"
                ".longitud"
            ),
            pedido.longitud,
        )

        if (
            pedido
            .tiene_ventana_especifica
        ):
            hora_desde = (
                pedido.hora_desde_min
            )

            hora_hasta = (
                pedido.hora_hasta_min
            )

            if (
                hora_desde
                < instancia
                .hora_inicio_turno_min
                or hora_hasta
                > instancia
                .hora_fin_objetivo_min
                or hora_desde
                >= hora_hasta
            ):
                raise ErrorContratoAlpyne(
                    f"Pedido {pedido.pedido_id}: "
                    "la ventana específica queda "
                    "fuera del horario normal "
                    "del turno."
                )

        else:
            hora_desde = -1
            hora_hasta = -1

        vector.extend(
            [
                float(
                    indice
                ),
                float(
                    original_indice
                ),
                float(
                    pedido.numero_parte
                ),
                float(
                    pedido.total_partes
                ),
                float(
                    pedido
                    .unidades_capacidad
                ),
                (
                    1.0
                    if pedido
                    .requiere_volcador
                    else 0.0
                ),
                float(
                    pedido.latitud
                ),
                float(
                    pedido.longitud
                ),
                float(
                    hora_desde
                ),
                float(
                    hora_hasta
                ),
            ]
        )

    longitud_esperada = (
        TAMANO_CABECERA_INSTANCIA
        +
        cantidad_pedidos
        * TAMANO_PEDIDO_INSTANCIA
    )

    if len(vector) != longitud_esperada:
        raise AssertionError(
            "Error interno al construir "
            "instanciaVector."
        )

    return vector


def codificar_plan_alpyne(
    instancia: InstanciaTurno,
    plan: PlanTurno,
) -> list[float]:
    """
    Convierte un PlanTurno al vector double[] del
    protocolo Alpyne v1 utilizado por AnyLogic.
    """
    validacion = validar_plan(
        instancia,
        plan,
    )

    if not validacion.valido:
        raise ErrorContratoAlpyne(
            "El plan es inválido: "
            + " | ".join(
                validacion.errores
            )
        )

    _exigir_finito_no_negativo(
        "plan.costo_estimado",
        plan.costo_estimado,
    )

    _exigir_finito_no_negativo(
        "plan.tiempo_computo_ms",
        plan.tiempo_computo_ms,
    )

    try:
        algoritmo_codigo = (
            CODIGO_ALGORITMO[
                plan.algoritmo
            ]
        )

    except KeyError as exc:
        raise ErrorContratoAlpyne(
            "Algoritmo no soportado por "
            "el protocolo Alpyne: "
            f"{plan.algoritmo!r}."
        ) from exc

    indice_por_pedido = {
        pedido.pedido_id: indice
        for indice, pedido in enumerate(
            instancia.pedidos
        )
    }

    asignaciones: list[
        tuple[int, int, int, int]
    ] = []

    camiones_ordenados = sorted(
        plan.camiones,
        key=lambda camion: (
            camion.camion_id
        ),
    )

    for plan_camion in camiones_ordenados:
        for numero_esperado, viaje in enumerate(
            plan_camion.viajes,
            start=1,
        ):
            if (
                viaje.numero_viaje
                != numero_esperado
            ):
                raise ErrorContratoAlpyne(
                    "Los viajes del camión "
                    f"{plan_camion.camion_id} "
                    "deben estar numerados de "
                    "forma consecutiva desde 1. "
                    f"Esperado={numero_esperado}, "
                    "recibido="
                    f"{viaje.numero_viaje}."
                )

            for orden, pedido_id in enumerate(
                viaje.pedido_ids,
                start=1,
            ):
                try:
                    pedido_indice = (
                        indice_por_pedido[
                            pedido_id
                        ]
                    )

                except KeyError as exc:
                    raise ErrorContratoAlpyne(
                        "El plan referencia un "
                        "pedido inexistente: "
                        f"{pedido_id}."
                    ) from exc

                asignaciones.append(
                    (
                        plan_camion.camion_id,
                        viaje.numero_viaje,
                        orden,
                        pedido_indice,
                    )
                )

    if len(asignaciones) != len(
        instancia.pedidos
    ):
        raise ErrorContratoAlpyne(
            "La cantidad de asignaciones no "
            "coincide con la cantidad de pedidos."
        )

    vector: list[float] = [
        float(
            PROTOCOL_VERSION
        ),
        float(
            len(
                asignaciones
            )
        ),
        float(
            algoritmo_codigo
        ),
        float(
            plan.costo_estimado
        ),
        float(
            plan.tiempo_computo_ms
        ),
    ]

    for (
        camion_id,
        numero_viaje,
        orden,
        pedido_indice,
    ) in asignaciones:
        vector.extend(
            [
                float(
                    camion_id
                ),
                float(
                    numero_viaje
                ),
                float(
                    orden
                ),
                float(
                    pedido_indice
                ),
            ]
        )

    longitud_esperada = (
        TAMANO_CABECERA_PLAN
        +
        len(
            asignaciones
        )
        * TAMANO_ASIGNACION_PLAN
    )

    if len(vector) != longitud_esperada:
        raise AssertionError(
            "Error interno al construir "
            "planVector."
        )

    return vector


def _exigir_finito_no_negativo(
    campo: str,
    valor: float,
) -> None:
    valor_float = float(
        valor
    )

    if not isfinite(
        valor_float
    ):
        raise ErrorContratoAlpyne(
            f"{campo} debe ser finito."
        )

    if valor_float < 0.0:
        raise ErrorContratoAlpyne(
            f"{campo} no puede ser negativo."
        )


def _exigir_entero_positivo(
    campo: str,
    valor: int,
) -> None:
    if not isinstance(
        valor,
        int,
    ):
        raise ErrorContratoAlpyne(
            f"{campo} debe ser int."
        )

    if valor <= 0:
        raise ErrorContratoAlpyne(
            f"{campo} debe ser mayor que cero."
        )


def _exigir_latitud(
    campo: str,
    valor: float,
) -> None:
    valor_float = float(
        valor
    )

    if (
        not isfinite(
            valor_float
        )
        or valor_float < -90.0
        or valor_float > 90.0
    ):
        raise ErrorContratoAlpyne(
            f"{campo} contiene una "
            "latitud inválida."
        )


def _exigir_longitud(
    campo: str,
    valor: float,
) -> None:
    valor_float = float(
        valor
    )

    if (
        not isfinite(
            valor_float
        )
        or valor_float < -180.0
        or valor_float > 180.0
    ):
        raise ErrorContratoAlpyne(
            f"{campo} contiene una "
            "longitud inválida."
        )