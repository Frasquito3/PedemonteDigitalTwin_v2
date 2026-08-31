from __future__ import annotations

import atexit
import base64
import os
import shutil
import time
import warnings
from dataclasses import asdict, dataclass
from math import isclose, isfinite
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


ACCION_EJECUTAR_PLAN_DINAMICO = 2
VERSION_RESULTADO_SIMULADO = 1
HORIZONTE_SIMULACION_MIN_PREDETERMINADO = 600.0


@dataclass(frozen=True)
class ResultadoEjecucionSimulada:
    version: int
    instancia_id: str
    fecha_operacion: str
    algoritmo_aplicado: str
    seed_escenario: int
    seed_ejecucion: int
    costo_total: float
    tareas_entregadas: int
    tareas_no_entregadas: int
    viajes_totales: int
    duracion_simulada_min: float
    distancia_total_km: float
    tardanza_total_min: float
    diferencia_fin_camiones_min: float
    ocupacion_global_pct: float
    costo_tareas_no_entregadas: float
    costo_pedidos_originales_incompletos: float
    costo_tardanza: float
    costo_exceso_tolerancia: float
    costo_operacion: float
    costo_distancia: float
    costo_viajes: float
    costo_desbalance: float
    mensaje: str
    estado_motor: str

    def como_dict(self) -> dict[str, Any]:
        return asdict(self)

    def resumen(self) -> str:
        return (
            "OK"
            f"|version={self.version}"
            f"|instancia={self.instancia_id}"
            f"|fecha={self.fecha_operacion}"
            f"|algoritmo={self.algoritmo_aplicado}"
            f"|seed_escenario={self.seed_escenario}"
            f"|seed_ejecucion={self.seed_ejecucion}"
            f"|costo={self.costo_total:.6f}"
            f"|distancia_km={self.distancia_total_km:.6f}"
            f"|duracion_min={self.duracion_simulada_min:.6f}"
            f"|viajes={self.viajes_totales}"
            f"|tardanza_min={self.tardanza_total_min:.6f}"
            f"|desbalance_min={self.diferencia_fin_camiones_min:.6f}"
        )


class ErrorEjecucionSimulada(RuntimeError):
    pass


def ejecutar_plan_en_modelo_exportado(
    *,
    modelo_exportado: str | Path,
    raiz_python: str | Path,
    instancia_vector: Sequence[float],
    plan_vector: Sequence[float],
    identificadores_pedidos: str,
    instancia_id: str,
    fecha_operacion: str,
    seed_escenario: int,
    seed_ejecucion: int,
    proveedores_habilitados: bool,
    timeout_segundos: int = 240,
    horizonte_simulacion_min: float = HORIZONTE_SIMULACION_MIN_PREDETERMINADO,
    log_id: str = "simulated-execution",
    java_exe: str | Path | None = None,
    sim_factory: Callable[..., Any] | None = None,
) -> ResultadoEjecucionSimulada:
    """Ejecuta un plan en una corrida limpia del modelo AnyLogic exportado."""
    modelo = Path(modelo_exportado).expanduser().resolve()
    raiz = Path(raiz_python).expanduser().resolve()
    ids_limpios = str(identificadores_pedidos).strip()
    id_limpio = str(instancia_id).strip()
    fecha_limpia = str(fecha_operacion).strip()

    if not modelo.is_file():
        raise FileNotFoundError(
            f"No existe el modelo AnyLogic exportado: {modelo}"
        )

    if not raiz.is_dir():
        raise NotADirectoryError(
            f"No existe la raíz Python del proyecto: {raiz}"
        )

    if not ids_limpios:
        raise ValueError("identificadores_pedidos no puede estar vacío.")

    if not id_limpio:
        raise ValueError("instancia_id no puede estar vacío.")

    if not fecha_limpia:
        raise ValueError("fecha_operacion no puede estar vacía.")

    if int(seed_escenario) <= 0:
        raise ValueError("seed_escenario debe ser > 0.")

    if int(seed_ejecucion) <= 0:
        raise ValueError("seed_ejecucion debe ser > 0.")

    instancia = _vector_finito_no_vacio(
        instancia_vector,
        "instancia_vector",
    )
    plan = _vector_finito_no_vacio(
        plan_vector,
        "plan_vector",
    )

    if timeout_segundos <= 0:
        raise ValueError("timeout_segundos debe ser > 0.")

    horizonte = float(horizonte_simulacion_min)

    if not isfinite(horizonte) or horizonte <= 0.0:
        raise ValueError(
            "horizonte_simulacion_min debe ser finito y > 0."
        )

    kwargs_sim: dict[str, Any] = {
        "auto_lock": True,
        "auto_finish": True,
        "lock_defaults": {"timeout": timeout_segundos},
        "engine_overrides": {"stop_time": horizonte},
        "py_log_level": True,
        "java_log_level": True,
        "log_id": log_id,
        "startup_delay": 1.0,
    }

    if sim_factory is None:
        try:
            from alpyne.sim import AnyLogicSim
        except ImportError as exc:  # pragma: no cover - depende del entorno
            raise RuntimeError(
                "No está instalado anylogic-alpyne. "
                "Instale requirements-dev.txt en el entorno virtual."
            ) from exc

        sim_factory = AnyLogicSim
        java_resuelto = _resolver_java(java_exe)
        kwargs_sim["java_exe"] = str(java_resuelto)
    elif java_exe is not None:
        kwargs_sim["java_exe"] = str(
            Path(java_exe).expanduser().resolve()
        )

    sim = None

    try:
        sim = sim_factory(
            str(modelo),
            **kwargs_sim,
        )

        estado_inicial = sim.reset(
            instanciaId=id_limpio,
            fechaOperacion=fecha_limpia,
            identificadoresPedidos=ids_limpios,
            seedEscenario=int(seed_escenario),
            seedEjecucion=int(seed_ejecucion),
            instanciaVector=instancia,
            rutaPythonProyectoPypeline=str(raiz),
            proveedoresHabilitados=bool(proveedores_habilitados),
        )

        if estado_inicial is None:
            raise ErrorEjecucionSimulada(
                "Alpyne no devolvió estado después de reset()."
            )

        observacion_inicial = _a_dict(
            estado_inicial.observation
        )

        if bool(observacion_inicial.get("error", False)):
            raise ErrorEjecucionSimulada(
                "El modelo informó un error durante la configuración: "
                + _normalizar_texto(
                    observacion_inicial.get("mensaje", "sin detalle")
                )
            )

        if not bool(observacion_inicial.get("configurado", False)):
            raise ErrorEjecucionSimulada(
                "El modelo no confirmó la configuración Alpyne."
            )

        estado_accion = sim.take_action(
            accionCodigo=ACCION_EJECUTAR_PLAN_DINAMICO,
            planVector=plan,
        )

        if estado_accion is None:
            raise ErrorEjecucionSimulada(
                "Alpyne no devolvió estado después de take_action()."
            )

        estado_final = _esperar_resultado_terminal(
            sim,
            estado_inicial=estado_inicial,
            estado_accion=estado_accion,
            timeout_segundos=timeout_segundos,
        )

        observacion_final = _a_dict(
            estado_final.observation
        )

        resultado = construir_resultado_simulado(
            observacion_final,
            estado_motor=_nombre_estado(
                estado_final.state
            ),
            stop=bool(estado_final.stop),
        )

        _validar_identidad_resultado(
            resultado,
            instancia_id=id_limpio,
            fecha_operacion=fecha_limpia,
            seed_escenario=int(seed_escenario),
            seed_ejecucion=int(seed_ejecucion),
        )

        return resultado

    finally:
        if sim is not None:
            _cerrar_simulacion(sim)



def _esperar_resultado_terminal(
    sim: Any,
    *,
    estado_inicial: Any,
    estado_accion: Any,
    timeout_segundos: int,
    intervalo_sondeo_segundos: float = 0.05,
) -> Any:
    """
    Espera una observación final funcional después de enviar la acción.

    ``AnyLogicSim.take_action()`` utiliza internamente un bloqueo sobre
    estados "ready", que incluye PAUSED. En algunas corridas rápidas puede
    devolver el PAUSED anterior a la acción antes de que el servidor publique
    el nuevo estado FINISHED. Por eso no se valida inmediatamente el primer
    objeto recibido: se consulta ``status()`` hasta observar una secuencia
    posterior con resultado completo.
    """
    if timeout_segundos <= 0:
        raise ValueError("timeout_segundos debe ser > 0.")

    if intervalo_sondeo_segundos <= 0.0:
        raise ValueError(
            "intervalo_sondeo_segundos debe ser > 0."
        )

    secuencia_inicial = _atributo_entero_opcional(
        estado_inicial,
        "sequence_id",
    )

    ultimo_estado = estado_accion
    limite = time.monotonic() + float(timeout_segundos)

    while True:
        observacion = _a_dict(
            getattr(
                ultimo_estado,
                "observation",
                {},
            )
        )

        estado_motor = _nombre_estado(
            getattr(
                ultimo_estado,
                "state",
                "DESCONOCIDO",
            )
        ).strip().upper()

        secuencia_actual = _atributo_entero_opcional(
            ultimo_estado,
            "sequence_id",
        )

        secuencia_avanzada = (
            secuencia_inicial is None
            or secuencia_actual is None
            or secuencia_actual > secuencia_inicial
        )

        ejecucion_finalizada = bool(
            observacion.get(
                "ejecucionFinalizada",
                False,
            )
        )

        resultado_disponible = bool(
            observacion.get(
                "resultadoDisponible",
                False,
            )
        )

        if (
            bool(
                observacion.get(
                    "error",
                    False,
                )
            )
            or estado_motor == "ERROR"
        ):
            return ultimo_estado

        if (
            secuencia_avanzada
            and ejecucion_finalizada
            and resultado_disponible
        ):
            return ultimo_estado

        if time.monotonic() >= limite:
            raise ErrorEjecucionSimulada(
                "La corrida no publicó un resultado terminal "
                "dentro del tiempo de espera: "
                f"estado={estado_motor or 'DESCONOCIDO'}, "
                f"secuencia_inicial={secuencia_inicial}, "
                f"secuencia_actual={secuencia_actual}, "
                "ejecucionFinalizada="
                f"{ejecucion_finalizada}, "
                "resultadoDisponible="
                f"{resultado_disponible}."
            )

        consultar_estado = getattr(
            sim,
            "status",
            None,
        )

        if not callable(consultar_estado):
            raise ErrorEjecucionSimulada(
                "El comunicador Alpyne no permite consultar "
                "el estado posterior a la acción."
            )

        time.sleep(
            intervalo_sondeo_segundos
        )

        estado_consultado = consultar_estado()

        if estado_consultado is not None:
            ultimo_estado = estado_consultado


def codificar_identificadores_pedidos(
    pedidos: Sequence[Any],
) -> str:
    partes = ["IDMAP1", str(len(pedidos))]

    for indice, pedido in enumerate(pedidos):
        pedido_id = str(
            getattr(pedido, "pedido_id", "")
        ).strip()
        original_id = str(
            getattr(pedido, "pedido_original_id", "")
        ).strip() or pedido_id

        if not pedido_id:
            raise ValueError(
                f"pedido[{indice}].pedido_id no puede estar vacío."
            )

        if not original_id:
            raise ValueError(
                f"pedido[{indice}].pedido_original_id no puede estar vacío."
            )

        tarea_b64 = base64.b64encode(
            pedido_id.encode("utf-8")
        ).decode("ascii")
        original_b64 = base64.b64encode(
            original_id.encode("utf-8")
        ).decode("ascii")
        partes.append(f"{tarea_b64}:{original_b64}")

    return "|".join(partes)

def construir_resultado_simulado(
    observacion: Mapping[str, Any],
    *,
    estado_motor: str,
    stop: bool,
) -> ResultadoEjecucionSimulada:
    if bool(observacion.get("error", False)):
        raise ErrorEjecucionSimulada(
            "AnyLogic informó un error: "
            + _normalizar_texto(
                observacion.get("mensaje", "sin detalle")
            )
        )

    ejecucion_finalizada = bool(
        observacion.get("ejecucionFinalizada", False)
    )
    resultado_disponible = bool(
        observacion.get("resultadoDisponible", False)
    )
    estado_normalizado = _normalizar_texto(
        estado_motor
    ).strip().upper()

    # El resultado funcional de AnyLogic es la fuente de verdad.
    # ``stop`` y el estado del motor se conservan en el resultado para
    # diagnóstico, pero pueden llegar desfasados respecto de la observación.

    if not ejecucion_finalizada:
        raise ErrorEjecucionSimulada(
            "AnyLogic no marcó la ejecución como finalizada."
        )

    if not resultado_disponible:
        raise ErrorEjecucionSimulada(
            "La observación final no contiene un resultado disponible."
        )

    if not bool(observacion.get("instanciaAceptada", False)):
        raise ErrorEjecucionSimulada(
            "AnyLogic no aceptó la instancia enviada."
        )

    if not bool(observacion.get("planAceptado", False)):
        raise ErrorEjecucionSimulada(
            "AnyLogic no aceptó el plan enviado."
        )

    costo_total = _numero(
        observacion,
        "costoTotal",
    )

    componentes = [
        _numero(observacion, "costoTareasNoEntregadas"),
        _numero(observacion, "costoPedidosOriginalesIncompletos"),
        _numero(observacion, "costoTardanza"),
        _numero(observacion, "costoExcesoTolerancia"),
        _numero(observacion, "costoOperacion"),
        _numero(observacion, "costoDistancia"),
        _numero(observacion, "costoViajes"),
        _numero(observacion, "costoDesbalance"),
    ]

    suma_componentes = sum(componentes)

    if not isclose(
        costo_total,
        suma_componentes,
        rel_tol=1e-9,
        abs_tol=1e-6,
    ):
        raise ErrorEjecucionSimulada(
            "El costo total no coincide con su desglose: "
            f"total={costo_total}, suma={suma_componentes}."
        )

    ocupacion_pct = _numero(
        observacion,
        "ocupacionGlobalPct",
    )

    if ocupacion_pct > 100.0 + 1e-9:
        raise ErrorEjecucionSimulada(
            "ocupacionGlobalPct no puede superar 100: "
            f"{ocupacion_pct}."
        )

    return ResultadoEjecucionSimulada(
        version=VERSION_RESULTADO_SIMULADO,
        instancia_id=_normalizar_texto(
            observacion.get("instanciaId", "")
        ),
        fecha_operacion=_normalizar_texto(
            observacion.get("fechaOperacionResultado", "")
        ),
        algoritmo_aplicado=_normalizar_texto(
            observacion.get("algoritmoAplicado", "")
        ),
        seed_escenario=_entero(
            observacion,
            "seedEscenarioResultado",
        ),
        seed_ejecucion=_entero(
            observacion,
            "seedEjecucionResultado",
        ),
        costo_total=costo_total,
        tareas_entregadas=_entero(
            observacion,
            "tareasEntregadas",
        ),
        tareas_no_entregadas=_entero(
            observacion,
            "tareasNoEntregadas",
        ),
        viajes_totales=_entero(
            observacion,
            "viajesTotales",
        ),
        duracion_simulada_min=_numero(
            observacion,
            "tiempoSimuladoMin",
        ),
        distancia_total_km=_numero(
            observacion,
            "distanciaTotalKm",
        ),
        tardanza_total_min=_numero(
            observacion,
            "tardanzaTotalMin",
        ),
        diferencia_fin_camiones_min=_numero(
            observacion,
            "diferenciaFinCamionesMin",
        ),
        ocupacion_global_pct=ocupacion_pct,
        costo_tareas_no_entregadas=componentes[0],
        costo_pedidos_originales_incompletos=componentes[1],
        costo_tardanza=componentes[2],
        costo_exceso_tolerancia=componentes[3],
        costo_operacion=componentes[4],
        costo_distancia=componentes[5],
        costo_viajes=componentes[6],
        costo_desbalance=componentes[7],
        mensaje=_normalizar_texto(
            observacion.get("mensaje", "")
        ),
        estado_motor=_normalizar_texto(estado_motor),
    )


def _cerrar_simulacion(sim: Any) -> None:
    """Cierra el proceso Java y elimina su carpeta temporal."""
    cerrar = getattr(sim, "_quit_app", None)

    if not callable(cerrar):
        return

    try:
        atexit.unregister(cerrar)
    except Exception:
        pass

    try:
        cerrar()
    except Exception as exc:
        warnings.warn(
            "No fue posible cerrar limpiamente el motor AnyLogic: "
            f"{type(exc).__name__}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )


def _validar_identidad_resultado(
    resultado: ResultadoEjecucionSimulada,
    *,
    instancia_id: str,
    fecha_operacion: str,
    seed_escenario: int,
    seed_ejecucion: int,
) -> None:
    diferencias: list[str] = []

    if resultado.instancia_id != instancia_id:
        diferencias.append(
            "instanciaId="
            f"{resultado.instancia_id!r} != {instancia_id!r}"
        )

    if resultado.fecha_operacion != fecha_operacion:
        diferencias.append(
            "fechaOperacion="
            f"{resultado.fecha_operacion!r} != {fecha_operacion!r}"
        )

    if resultado.seed_escenario != seed_escenario:
        diferencias.append(
            "seedEscenario="
            f"{resultado.seed_escenario} != {seed_escenario}"
        )

    if resultado.seed_ejecucion != seed_ejecucion:
        diferencias.append(
            "seedEjecucion="
            f"{resultado.seed_ejecucion} != {seed_ejecucion}"
        )

    if diferencias:
        raise ErrorEjecucionSimulada(
            "La corrida no preservó la identidad de la instancia: "
            + " | ".join(diferencias)
        )


def _resolver_java(
    java_exe: str | Path | None,
) -> Path:
    if java_exe is not None:
        explicito = Path(java_exe).expanduser().resolve()
        if not explicito.is_file():
            raise FileNotFoundError(
                f"No existe java.exe en la ruta indicada: {explicito}"
            )
        return explicito

    en_path = shutil.which("java")
    if en_path:
        return Path(en_path).resolve()

    if os.name == "nt":
        candidatos: list[Path] = []
        for variable in ("ProgramFiles", "ProgramFiles(x86)"):
            raiz_texto = os.environ.get(variable)
            if not raiz_texto:
                continue
            raiz = Path(raiz_texto)
            for anylogic in sorted(raiz.glob("AnyLogic*")):
                candidatos.extend(
                    [
                        anylogic / "jre" / "bin" / "java.exe",
                        anylogic / "jdk" / "bin" / "java.exe",
                    ]
                )

        for candidato in candidatos:
            if candidato.is_file():
                return candidato.resolve()

    raise FileNotFoundError(
        "No se encontró Java. Indique java_exe o utilice "
        "el JRE incluido con AnyLogic."
    )


def _vector_finito_no_vacio(
    valores: Sequence[float],
    nombre: str,
) -> list[float]:
    if valores is None:
        raise ValueError(f"{nombre} no puede ser null.")

    vector = [float(valor) for valor in valores]

    if not vector:
        raise ValueError(f"{nombre} no puede estar vacío.")

    if not all(isfinite(valor) for valor in vector):
        raise ValueError(f"{nombre} debe contener solo valores finitos.")

    return vector


def _a_dict(valor: Any) -> dict[str, Any]:
    try:
        return dict(valor)
    except Exception as exc:
        raise ErrorEjecucionSimulada(
            "No fue posible convertir la observación a un diccionario."
        ) from exc


def _numero(
    observacion: Mapping[str, Any],
    campo: str,
) -> float:
    if campo not in observacion:
        raise ErrorEjecucionSimulada(
            f"La observación no contiene el campo {campo}."
        )

    valor = float(observacion[campo])

    if not isfinite(valor):
        raise ErrorEjecucionSimulada(
            f"El campo {campo} no es finito: {valor}."
        )

    if valor < 0.0:
        raise ErrorEjecucionSimulada(
            f"El campo {campo} no puede ser negativo: {valor}."
        )

    return valor


def _entero(
    observacion: Mapping[str, Any],
    campo: str,
) -> int:
    valor = _numero(observacion, campo)
    redondeado = round(valor)

    if not isclose(valor, redondeado, abs_tol=1e-9):
        raise ErrorEjecucionSimulada(
            f"El campo {campo} debe ser entero: {valor}."
        )

    return int(redondeado)


def _atributo_entero_opcional(
    objeto: Any,
    nombre: str,
) -> int | None:
    valor = getattr(
        objeto,
        nombre,
        None,
    )

    if valor is None:
        return None

    try:
        return int(valor)
    except (TypeError, ValueError):
        return None


def _nombre_estado(estado: Any) -> str:
    nombre = getattr(estado, "name", None)
    return str(nombre if nombre is not None else estado)


def _normalizar_texto(valor: Any) -> str:
    texto = str(valor)

    if "Ã" not in texto and "Â" not in texto:
        return texto

    try:
        return texto.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return texto
