from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

from planner.integration.estimated_comparison import (
    firmar_instancia_vector,
)
from planner.integration.simulated_execution import (
    ResultadoEjecucionSimulada,
    ejecutar_plan_en_modelo_exportado,
)


VERSION_COMPARACION_SIMULADA = 1

METODOS_COMPARACION_SIMULADA: tuple[str, ...] = (
    "RL",
    "HIBRIDO",
    "GREEDY",
    "RANDOM",
    "GA",
)

ESTADO_FINALIZADO = "FINALIZADO"
ESTADO_ERROR = "ERROR"
ESTADO_OMITIDO = "OMITIDO"


@dataclass(frozen=True)
class ResultadoMetodoSimulado:
    metodo_solicitado: str
    estado: str
    resultado: ResultadoEjecucionSimulada | None
    error: str
    tiempo_motor_segundos: float

    @property
    def finalizado(self) -> bool:
        return (
            self.estado == ESTADO_FINALIZADO
            and self.resultado is not None
        )

    def como_dict(self) -> dict[str, Any]:
        return {
            "metodo_solicitado": self.metodo_solicitado,
            "estado": self.estado,
            "error": self.error,
            "tiempo_motor_segundos": self.tiempo_motor_segundos,
            "resultado": (
                None
                if self.resultado is None
                else self.resultado.como_dict()
            ),
        }


@dataclass(frozen=True)
class ComparacionSimulada:
    version: int
    instancia_id: str
    fecha_operacion: str
    firma_instancia: str
    seed_escenario: int
    seed_ejecucion: int
    proveedores_habilitados: bool
    resultados: tuple[ResultadoMetodoSimulado, ...]
    tiempo_total_segundos: float

    @property
    def cantidad_finalizados(self) -> int:
        return sum(
            1
            for resultado in self.resultados
            if resultado.finalizado
        )

    @property
    def completa(self) -> bool:
        return (
            len(self.resultados)
            == len(METODOS_COMPARACION_SIMULADA)
            and self.cantidad_finalizados
            == len(METODOS_COMPARACION_SIMULADA)
        )

    def obtener_resultado(
        self,
        metodo: str,
    ) -> ResultadoMetodoSimulado:
        normalizado = normalizar_metodo_simulado(metodo)

        for resultado in self.resultados:
            if resultado.metodo_solicitado == normalizado:
                return resultado

        raise KeyError(
            "La comparación simulada no contiene el método "
            f"{normalizado}."
        )

    def como_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "instancia_id": self.instancia_id,
            "fecha_operacion": self.fecha_operacion,
            "firma_instancia": self.firma_instancia,
            "seed_escenario": self.seed_escenario,
            "seed_ejecucion": self.seed_ejecucion,
            "proveedores_habilitados": self.proveedores_habilitados,
            "completa": self.completa,
            "cantidad_finalizados": self.cantidad_finalizados,
            "cantidad_metodos": len(self.resultados),
            "tiempo_total_segundos": self.tiempo_total_segundos,
            "resultados": [
                resultado.como_dict()
                for resultado in self.resultados
            ],
        }

    def resumen(self) -> str:
        errores = sum(
            1
            for resultado in self.resultados
            if resultado.estado == ESTADO_ERROR
        )
        omitidos = sum(
            1
            for resultado in self.resultados
            if resultado.estado == ESTADO_OMITIDO
        )

        return (
            "OK"
            f"|version={self.version}"
            f"|instancia={self.instancia_id}"
            f"|firma={self.firma_instancia}"
            f"|finalizados={self.cantidad_finalizados}"
            f"|errores={errores}"
            f"|omitidos={omitidos}"
            f"|completa={'SI' if self.completa else 'NO'}"
            f"|tiempo_total_s={self.tiempo_total_segundos:.6f}"
        )


RunnerSimulado = Callable[..., ResultadoEjecucionSimulada]


def ejecutar_comparacion_simulada(
    *,
    modelo_exportado: str | Path,
    raiz_python: str | Path,
    instancia_vector: Sequence[float],
    planes_por_metodo: Mapping[str, Sequence[float]],
    identificadores_pedidos: str,
    instancia_id: str,
    fecha_operacion: str,
    seed_escenario: int,
    seed_ejecucion: int,
    proveedores_habilitados: bool,
    timeout_segundos_por_metodo: int = 240,
    horizonte_simulacion_min: float = 600.0,
    continuar_ante_error: bool = True,
    runner: RunnerSimulado = ejecutar_plan_en_modelo_exportado,
) -> ComparacionSimulada:
    """
    Ejecuta cada método en un motor AnyLogic nuevo e independiente.

    Todos los métodos comparten la misma identidad y la misma semilla de
    ejecución para aplicar números aleatorios comunes. Cada llamada al runner
    crea y cierra su propio proceso Java.
    """
    if timeout_segundos_por_metodo <= 0:
        raise ValueError(
            "timeout_segundos_por_metodo debe ser > 0."
        )

    horizonte = float(horizonte_simulacion_min)

    if not isfinite(horizonte) or horizonte <= 0.0:
        raise ValueError(
            "horizonte_simulacion_min debe ser finito y > 0."
        )

    vector_instancia = _vector_finito_no_vacio(
        instancia_vector,
        "instancia_vector",
    )
    planes = _normalizar_planes(planes_por_metodo)

    firma = firmar_instancia_vector(
        vector_instancia,
        int(seed_escenario),
        int(seed_ejecucion),
    )

    inicio_total = perf_counter()
    resultados: list[ResultadoMetodoSimulado] = []

    for metodo in METODOS_COMPARACION_SIMULADA:
        plan = planes.get(metodo)

        if plan is None:
            resultados.append(
                ResultadoMetodoSimulado(
                    metodo_solicitado=metodo,
                    estado=ESTADO_OMITIDO,
                    resultado=None,
                    error="No existe un plan factible almacenado.",
                    tiempo_motor_segundos=0.0,
                )
            )
            continue

        inicio_metodo = perf_counter()

        try:
            resultado = runner(
                modelo_exportado=modelo_exportado,
                raiz_python=raiz_python,
                instancia_vector=vector_instancia,
                plan_vector=plan,
                identificadores_pedidos=identificadores_pedidos,
                instancia_id=instancia_id,
                fecha_operacion=fecha_operacion,
                seed_escenario=int(seed_escenario),
                seed_ejecucion=int(seed_ejecucion),
                proveedores_habilitados=bool(
                    proveedores_habilitados
                ),
                timeout_segundos=timeout_segundos_por_metodo,
                horizonte_simulacion_min=horizonte,
                log_id=(
                    "simulated-comparison-"
                    + metodo.lower()
                ),
            )

            resultados.append(
                ResultadoMetodoSimulado(
                    metodo_solicitado=metodo,
                    estado=ESTADO_FINALIZADO,
                    resultado=resultado,
                    error="",
                    tiempo_motor_segundos=(
                        perf_counter() - inicio_metodo
                    ),
                )
            )

        except Exception as exc:
            resultados.append(
                ResultadoMetodoSimulado(
                    metodo_solicitado=metodo,
                    estado=ESTADO_ERROR,
                    resultado=None,
                    error=f"{type(exc).__name__}: {exc}",
                    tiempo_motor_segundos=(
                        perf_counter() - inicio_metodo
                    ),
                )
            )

            if not continuar_ante_error:
                break

    tiempo_total = perf_counter() - inicio_total

    return ComparacionSimulada(
        version=VERSION_COMPARACION_SIMULADA,
        instancia_id=str(instancia_id).strip(),
        fecha_operacion=str(fecha_operacion).strip(),
        firma_instancia=firma,
        seed_escenario=int(seed_escenario),
        seed_ejecucion=int(seed_ejecucion),
        proveedores_habilitados=bool(
            proveedores_habilitados
        ),
        resultados=tuple(resultados),
        tiempo_total_segundos=tiempo_total,
    )


def normalizar_metodo_simulado(metodo: str) -> str:
    normalizado = (
        str(metodo)
        .strip()
        .upper()
        .replace("Í", "I")
    )

    if normalizado not in METODOS_COMPARACION_SIMULADA:
        raise ValueError(
            "Método de comparación simulada no reconocido: "
            f"{metodo!r}."
        )

    return normalizado


def _normalizar_planes(
    planes_por_metodo: Mapping[str, Sequence[float]],
) -> dict[str, list[float]]:
    if planes_por_metodo is None:
        raise ValueError(
            "planes_por_metodo no puede ser null."
        )

    normalizados: dict[str, list[float]] = {}

    for metodo, plan in planes_por_metodo.items():
        clave = normalizar_metodo_simulado(metodo)

        if clave in normalizados:
            raise ValueError(
                "Se recibió más de un plan para el método "
                f"{clave}."
            )

        normalizados[clave] = _vector_finito_no_vacio(
            plan,
            f"plan[{clave}]",
        )

    return normalizados


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
        raise ValueError(
            f"{nombre} debe contener solo valores finitos."
        )

    return vector
