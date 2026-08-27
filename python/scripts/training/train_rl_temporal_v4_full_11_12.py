from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Any, Callable

# pyrefly: ignore [missing-import]
import gymnasium as gym

# pyrefly: ignore [missing-import]
from sb3_contrib import MaskablePPO

# pyrefly: ignore [missing-import]
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback

# pyrefly: ignore [missing-import]
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)

# pyrefly: ignore [missing-import]
from stable_baselines3.common.monitor import Monitor

# pyrefly: ignore [missing-import]
from stable_baselines3.common.vec_env import DummyVecEnv


PYTHON_ROOT = Path(__file__).resolve().parents[2]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))


from planner.rl.instance_generator import (  # noqa: E402
    ConfiguracionGeneradorInstancias,
    GeneradorInstanciasRL,
)
from planner.rl.rl_reward import (  # noqa: E402
    ConfiguracionRewardRL,
    ModoRewardRL,
)
from planner.rl.rl_temporal_v4_config import (  # noqa: E402
    ConfiguracionTemporalV4RL,
)
from planner.rl.rl_temporal_v4_full_curriculum import (  # noqa: E402
    TIMESTEPS_ACUMULADOS_MAXIMOS_V4,
    TIMESTEPS_BASE_EXTENSION_V4,
    TIMESTEPS_COMPLETOS_ADICIONALES_V4,
    EtapaEntrenamientoCompletoTemporalV4RL,
    crear_curriculum_entrenamiento_completo_temporal_v4,
)
from planner.rl.rl_temporal_v4_full_validation import (  # noqa: E402
    BateriaValidacionCompletaV4,
    ResumenValidacionCompletaV4,
    crear_bateria_validacion_completa_v4,
    es_mejor_validacion_completa_v4,
    evaluar_modelo_externamente_completo_v4,
    hash_archivo,
    semillas_por_estrato,
    validar_evidencia_holdout_16d7,
    validar_origen_extension_v4,
)
from planner.rl.rl_temporal_v4_training_env import (  # noqa: E402
    PedemonteTemporalV4TrainingEnv,
)
from planner.rl.temporal_v4_full_generator import (  # noqa: E402
    FuenteReplayTemporalV4,
    GeneradorMezclaCompletaTemporalV4RL,
)
from planner.rl.temporal_v4_instance_generator import (  # noqa: E402
    ConfiguracionGeneradorTemporalV4,
    GeneradorInstanciasTemporalV4RL,
)
from planner.routing.vial_cache import (  # noqa: E402
    ProveedorVialCachePersistente,
)


INFO_KEYWORDS = (
    "costo_estimado",
    "cantidad_viajes",
    "seed_instancia",
    "costo_greedy_referencia",
    "gap_relativo_greedy",
    "pedidos_tardios_prefijo",
    "tardanza_prefijo_min",
    "reward_arrepentimiento_local",
    "arrepentimiento_local_normalizado",
    "reward_terminal_v4",
    "componente_terminal_factibilidad",
    "componente_terminal_costo_acotado",
)


class ValidacionCompletaV4Callback(BaseCallback):
    def __init__(
        self,
        *,
        eval_freq: int,
        stage_dir: Path,
        bateria: BateriaValidacionCompletaV4,
        proveedor_clasicos: Any,
        configuracion_temporal: ConfiguracionTemporalV4RL,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        if eval_freq <= 0:
            raise ValueError("eval_freq debe ser > 0.")

        self.eval_freq = eval_freq
        self.stage_dir = stage_dir
        self.bateria = bateria
        self.proveedor_clasicos = proveedor_clasicos
        self.configuracion_temporal = configuracion_temporal
        self.external_dir = stage_dir / "external_validation"
        self.best_dir = stage_dir / "external_best"
        self.external_dir.mkdir(parents=True, exist_ok=True)
        self.best_dir.mkdir(parents=True, exist_ok=True)
        self.historial_csv = self.external_dir / "history.csv"
        self.best_summary: ResumenValidacionCompletaV4 | None = None

    @property
    def best_model_path(self) -> Path:
        return self.best_dir / "best_model.zip"

    def _guardar_historial(
        self,
        resumen: ResumenValidacionCompletaV4,
        mejorado: bool,
    ) -> None:
        fila = {
            "timestep": resumen.timestep,
            "clasicos_pedidos_tardios": (
                resumen.clasicos_pedidos_tardios
            ),
            "clasicos_tardanza_total_min": (
                resumen.clasicos_tardanza_total_min
            ),
            "clasicos_regresiones_costo": (
                resumen.clasicos_regresiones_costo
            ),
            "guard_3_8_sin_riesgo": resumen.guard_3_8_sin_riesgo,
            "guard_3_8_totales": resumen.guard_3_8_totales,
            "guard_3_8_pedidos_tardios": (
                resumen.guard_3_8_pedidos_tardios
            ),
            "guard_3_8_tardanza_total_min": (
                resumen.guard_3_8_tardanza_total_min
            ),
            "guard_9_10_sin_riesgo": resumen.guard_9_10_sin_riesgo,
            "guard_9_10_totales": resumen.guard_9_10_totales,
            "guard_9_10_pedidos_tardios": (
                resumen.guard_9_10_pedidos_tardios
            ),
            "guard_9_10_tardanza_total_min": (
                resumen.guard_9_10_tardanza_total_min
            ),
            "objetivo_11_sin_riesgo": resumen.objetivo_11_sin_riesgo,
            "objetivo_11_totales": resumen.objetivo_11_totales,
            "objetivo_11_pedidos_tardios": (
                resumen.objetivo_11_pedidos_tardios
            ),
            "objetivo_11_tardanza_total_min": (
                resumen.objetivo_11_tardanza_total_min
            ),
            "objetivo_12_sin_riesgo": resumen.objetivo_12_sin_riesgo,
            "objetivo_12_totales": resumen.objetivo_12_totales,
            "objetivo_12_pedidos_tardios": (
                resumen.objetivo_12_pedidos_tardios
            ),
            "objetivo_12_tardanza_total_min": (
                resumen.objetivo_12_tardanza_total_min
            ),
            "objetivo_general_11_12_sin_riesgo": (
                resumen.objetivo_general_11_12_sin_riesgo
            ),
            "objetivo_general_11_12_totales": (
                resumen.objetivo_general_11_12_totales
            ),
            "objetivo_11_12_sin_riesgo": (
                resumen.objetivo_11_12_sin_riesgo
            ),
            "objetivo_11_12_totales": resumen.objetivo_11_12_totales,
            "objetivo_11_12_costos_extremos": (
                resumen.objetivo_11_12_costos_extremos
            ),
            "gap_costo_mediano_vs_greedy_pct": (
                resumen.gap_costo_mediano_vs_greedy_pct
            ),
            "nuevo_mejor_externo": mejorado,
        }
        existe = self.historial_csv.exists()
        with self.historial_csv.open(
            "a",
            encoding="utf-8",
            newline="",
        ) as archivo:
            escritor = csv.DictWriter(archivo, fieldnames=list(fila))
            if not existe:
                escritor.writeheader()
            escritor.writerow(fila)

        guardar_json(
            self.external_dir / "latest.json",
            resumen.como_dict(),
        )

    def evaluar_modelo_y_guardar(
        self,
        model: Any,
        timestep: int,
    ) -> ResumenValidacionCompletaV4:
        resumen = evaluar_modelo_externamente_completo_v4(
            model,
            timestep=int(timestep),
            bateria=self.bateria,
            proveedor_clasicos=self.proveedor_clasicos,
            configuracion_temporal=self.configuracion_temporal,
        )
        mejorado = es_mejor_validacion_completa_v4(
            resumen,
            self.best_summary,
        )
        if mejorado:
            self.best_summary = resumen
            model.save(str(self.best_dir / "best_model"))
            guardar_json(
                self.best_dir / "best_summary.json",
                resumen.como_dict(),
            )

        self._guardar_historial(resumen, mejorado)
        if self.verbose:
            marca = "NUEVO_MEJOR" if mejorado else "sin_mejora"
            print(
                "External V4 FULL "
                f"t={resumen.timestep} | "
                "clásicos tardíos="
                f"{resumen.clasicos_pedidos_tardios} | "
                "guard 3-8="
                f"{resumen.guard_3_8_sin_riesgo}/"
                f"{resumen.guard_3_8_totales} | "
                "guard 9-10="
                f"{resumen.guard_9_10_sin_riesgo}/"
                f"{resumen.guard_9_10_totales} | "
                "exactos 12="
                f"{resumen.objetivo_12_sin_riesgo}/"
                f"{resumen.objetivo_12_totales} | "
                "general 11-12="
                f"{resumen.objetivo_general_11_12_sin_riesgo}/"
                f"{resumen.objetivo_general_11_12_totales} | "
                f"{marca}"
            )
        return resumen

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True
        self.evaluar_modelo_y_guardar(
            self.model,
            int(self.num_timesteps),
        )
        return True


def parse_args() -> argparse.Namespace:
    base_dir = (
        PYTHON_ROOT
        / "rl_artifacts"
        / "phase16d_temporal_v4_extension_9_12"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Continúa el checkpoint temporal v4 de 68.288 pasos con un "
            "entrenamiento completo enfocado en 11-12 y exactamente 12 "
            "pedidos, manteniendo replay de 3-8 y 9-10."
        )
    )
    parser.add_argument(
        "--base-model",
        type=Path,
        default=base_dir / "final_model.zip",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=base_dir / "temporal_v4_extension_config.json",
    )
    parser.add_argument(
        "--base-selection",
        type=Path,
        default=base_dir / "final_model_selection.json",
    )
    parser.add_argument(
        "--base-summary",
        type=Path,
        default=base_dir / "final_external_summary.json",
    )
    parser.add_argument(
        "--temporal-config",
        type=Path,
        default=(
            PYTHON_ROOT
            / "rl_artifacts"
            / "phase16d_temporal_v4_quick"
            / "temporal_v4_config.json"
        ),
        help=(
            "Configuración original que contiene los parámetros temporal v4. "
            "La extensión 9-12 conserva esos parámetros pero no los duplica."
        ),
    )
    parser.add_argument(
        "--holdout-evidence",
        type=Path,
        default=(
            PYTHON_ROOT
            / "results"
            / "rl_temporal"
            / "16D_7_holdout_extension_9_12_formal_272000"
            / "rl_temporal_v4_extension_holdout.json"
        ),
    )
    parser.add_argument(
        "--run-name",
        default="phase16d_temporal_v4_full_11_12",
    )
    parser.add_argument("--seed", type=int, default=168_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument(
        "--validation-cases-per-stratum",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--validation-seed-start",
        type=int,
        default=273_000,
    )
    parser.add_argument(
        "--internal-eval-seed-start",
        type=int,
        default=276_000,
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Valida archivos, carga el modelo y ejecuta la batería externa "
            "sin entrenar ni crear un run dentro de rl_artifacts."
        ),
    )
    parser.add_argument(
        "--preflight-output",
        type=Path,
        default=(
            PYTHON_ROOT
            / "results"
            / "rl_temporal"
            / "16D_8_preflight"
            / "preflight_summary.json"
        ),
    )
    args = parser.parse_args()

    if not args.run_name.strip():
        parser.error("--run-name no puede estar vacío.")
    if args.n_envs <= 0:
        parser.error("--n-envs debe ser > 0.")
    if args.validation_cases_per_stratum <= 0:
        parser.error("--validation-cases-per-stratum debe ser > 0.")
    if not 273_000 <= args.validation_seed_start < 274_000:
        parser.error(
            "--validation-seed-start debe estar entre 273000 y 273999."
        )
    if args.internal_eval_seed_start < 276_000:
        parser.error("--internal-eval-seed-start debe ser >= 276000.")
    return args


def _config_generador(
    *,
    min_pedidos: int,
    max_pedidos: int,
    probabilidad_ventana: float,
    probabilidad_volcador: float,
    probabilidad_pedido_grande: float,
    ancho_min: int,
    ancho_max: int,
) -> ConfiguracionGeneradorInstancias:
    return ConfiguracionGeneradorInstancias(
        min_pedidos_finales=min_pedidos,
        max_pedidos_finales=max_pedidos,
        probabilidad_volcador=probabilidad_volcador,
        probabilidad_ventana_especifica=probabilidad_ventana,
        probabilidad_pedido_mayor_capacidad=probabilidad_pedido_grande,
        ancho_ventana_min=ancho_min,
        ancho_ventana_max=ancho_max,
    )


def _generador_temporal(
    configuracion: ConfiguracionGeneradorInstancias,
    probabilidad_patron: float,
) -> GeneradorInstanciasTemporalV4RL:
    return GeneradorInstanciasTemporalV4RL(
        GeneradorInstanciasRL(configuracion),
        ConfiguracionGeneradorTemporalV4(
            probabilidad_patron_ventanas_conflictivas=(
                probabilidad_patron
            )
        ),
    )


def _generador_banda(
    *,
    minimo: int,
    maximo: int,
    probabilidad_patron: float,
    probabilidad_ventana: float = 0.90,
    probabilidad_volcador: float = 0.15,
    probabilidad_pedido_grande: float = 0.05,
    ancho_min: int = 45,
    ancho_max: int = 150,
) -> GeneradorInstanciasTemporalV4RL:
    return _generador_temporal(
        _config_generador(
            min_pedidos=minimo,
            max_pedidos=maximo,
            probabilidad_ventana=probabilidad_ventana,
            probabilidad_volcador=probabilidad_volcador,
            probabilidad_pedido_grande=probabilidad_pedido_grande,
            ancho_min=ancho_min,
            ancho_max=ancho_max,
        ),
        probabilidad_patron,
    )


def crear_generador_entrenamiento_completo(
    etapa: EtapaEntrenamientoCompletoTemporalV4RL,
) -> GeneradorMezclaCompletaTemporalV4RL:
    actual = _generador_banda(
        minimo=etapa.min_pedidos_finales,
        maximo=etapa.max_pedidos_finales,
        probabilidad_patron=etapa.probabilidad_patron_conflictivo,
        probabilidad_ventana=etapa.probabilidad_ventana_especifica,
        probabilidad_volcador=etapa.probabilidad_volcador,
        probabilidad_pedido_grande=etapa.probabilidad_pedido_grande,
        ancho_min=etapa.ancho_ventana_min,
        ancho_max=etapa.ancho_ventana_max,
    )
    replay_3_8 = _generador_banda(
        minimo=3,
        maximo=8,
        probabilidad_patron=0.50,
        probabilidad_volcador=0.10,
        probabilidad_pedido_grande=0.03,
    )
    replay_9_10 = _generador_banda(
        minimo=9,
        maximo=10,
        probabilidad_patron=0.50,
    )
    replay_general_11_12 = _generador_banda(
        minimo=11,
        maximo=12,
        probabilidad_patron=0.0,
    )
    replay_exactos_12 = _generador_banda(
        minimo=12,
        maximo=12,
        probabilidad_patron=0.50,
    )

    fuentes = tuple(
        fuente
        for fuente in (
            FuenteReplayTemporalV4(
                nombre="REPLAY_3_8",
                generador=replay_3_8,
                probabilidad=etapa.probabilidad_replay_3_8,
                mascara_seed=0x380000,
            ),
            FuenteReplayTemporalV4(
                nombre="REPLAY_9_10",
                generador=replay_9_10,
                probabilidad=etapa.probabilidad_replay_9_10,
                mascara_seed=0x910000,
            ),
            FuenteReplayTemporalV4(
                nombre="REPLAY_GENERAL_11_12",
                generador=replay_general_11_12,
                probabilidad=etapa.probabilidad_replay_general_11_12,
                mascara_seed=0x111200,
            ),
            FuenteReplayTemporalV4(
                nombre="REPLAY_EXACTOS_12",
                generador=replay_exactos_12,
                probabilidad=etapa.probabilidad_replay_exactos_12,
                mascara_seed=0x120000,
            ),
        )
        if fuente.probabilidad > 0.0
    )
    return GeneradorMezclaCompletaTemporalV4RL(
        generador_actual=actual,
        fuentes_replay=fuentes,
    )


def crear_env_factory(
    *,
    rank: int,
    seed_base: int,
    etapa: EtapaEntrenamientoCompletoTemporalV4RL,
    configuracion_reward: ConfiguracionRewardRL,
    configuracion_temporal: ConfiguracionTemporalV4RL,
    semillas_fijas: list[int] | None = None,
) -> Callable[[], gym.Env]:
    def _crear() -> gym.Env:
        env = PedemonteTemporalV4TrainingEnv(
            generador=crear_generador_entrenamiento_completo(etapa),
            seed_base=seed_base + rank * 100_000,
            semillas_fijas=semillas_fijas,
            max_pedidos=30,
            configuracion_reward=configuracion_reward,
            configuracion_temporal=configuracion_temporal,
        )
        return Monitor(env, info_keywords=INFO_KEYWORDS)

    return _crear


def validar_espacios_modelo(
    model: Any,
    *,
    timestep_esperado: int | None = None,
) -> None:
    forma = getattr(model.observation_space, "shape", None)
    acciones = getattr(model.action_space, "n", None)
    if forma != (702,):
        raise RuntimeError(
            "El modelo no corresponde al entorno temporal v4. "
            f"Observación encontrada: {forma}; esperada: (702,)."
        )
    if int(acciones) != 30:
        raise RuntimeError("El modelo no tiene las 30 acciones esperadas.")
    if timestep_esperado is not None and int(model.num_timesteps) != int(
        timestep_esperado
    ):
        raise RuntimeError(
            "El modelo base no tiene los timesteps esperados: "
            f"{model.num_timesteps} != {timestep_esperado}."
        )


def _normalizar_json(valor: Any) -> Any:
    if isinstance(valor, Enum):
        return valor.value
    if isinstance(valor, Path):
        return str(valor)
    if isinstance(valor, dict):
        return {
            str(clave): _normalizar_json(contenido)
            for clave, contenido in valor.items()
        }
    if isinstance(valor, (list, tuple)):
        return [_normalizar_json(item) for item in valor]
    return valor


def guardar_json(ruta: Path, contenido: dict[str, Any]) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(
        json.dumps(
            _normalizar_json(contenido),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )



def _imprimir_resumen_preflight(
    resumen: ResumenValidacionCompletaV4,
) -> None:
    print("Resumen de la batería externa fija:")
    print(
        "- B04-B06 tardíos/tardanza/regresiones costo: "
        f"{resumen.clasicos_pedidos_tardios}/"
        f"{resumen.clasicos_tardanza_total_min:.3f}/"
        f"{resumen.clasicos_regresiones_costo}"
    )
    print(
        "- Guard 3-8 sin riesgo: "
        f"{resumen.guard_3_8_sin_riesgo}/"
        f"{resumen.guard_3_8_totales}"
    )
    print(
        "- Guard 9-10 sin riesgo: "
        f"{resumen.guard_9_10_sin_riesgo}/"
        f"{resumen.guard_9_10_totales}"
    )
    print(
        "- Objetivo 11 sin riesgo: "
        f"{resumen.objetivo_11_sin_riesgo}/"
        f"{resumen.objetivo_11_totales}"
    )
    print(
        "- Objetivo exactos 12 sin riesgo: "
        f"{resumen.objetivo_12_sin_riesgo}/"
        f"{resumen.objetivo_12_totales}"
    )
    print(
        "- General 11-12 sin riesgo: "
        f"{resumen.objetivo_general_11_12_sin_riesgo}/"
        f"{resumen.objetivo_general_11_12_totales}"
    )


def main() -> None:
    args = parse_args()
    config_base, seleccion_base, resumen_base = validar_origen_extension_v4(
        args.base_config,
        args.base_selection,
        args.base_summary,
    )
    evidencia_holdout = validar_evidencia_holdout_16d7(
        args.holdout_evidence
    )
    if not args.base_model.is_file():
        raise FileNotFoundError(
            f"No existe el modelo base de 68.288 pasos: {args.base_model}"
        )

    if not args.temporal_config.is_file():
        raise FileNotFoundError(
            "No existe la configuración temporal v4 original: "
            f"{args.temporal_config}"
        )
    configuracion_original = json.loads(
        args.temporal_config.read_text(encoding="utf-8")
    )
    if not isinstance(configuracion_original, dict):
        raise ValueError("La configuración temporal debe ser un objeto JSON.")
    if configuracion_original.get("version_entorno") != (
        "pedemonte-rl-temporal-v4"
    ):
        raise ValueError("La configuración temporal v4 tiene versión inválida.")
    temporal_dict = configuracion_original.get("temporal")
    if not isinstance(temporal_dict, dict):
        raise ValueError("La configuración original no contiene temporal.")
    if temporal_dict.get("usar_mascara_temporal_dura") is not False:
        raise ValueError("La máscara temporal dura debe seguir desactivada.")
    configuracion_temporal = ConfiguracionTemporalV4RL(**temporal_dict)
    configuracion_reward = ConfiguracionRewardRL(
        modo=ModoRewardRL.VENTAJA_GREEDY_RELATIVA,
        denominador_relativo_minimo=1.0,
    )
    bateria = crear_bateria_validacion_completa_v4(
        cantidad_por_estrato=args.validation_cases_per_stratum,
        seed_inicio=args.validation_seed_start,
    )
    proveedor_clasicos = ProveedorVialCachePersistente(
        PYTHON_ROOT / "data" / "routing" / "cache_vial_v1.csv",
        version_cache_esperada="pedemonte-vial-v1",
        permitir_fallback=False,
    )

    if max(bateria.semillas_sinteticas, default=0) >= 274_000:
        raise RuntimeError(
            "La selección externa invadió el rango reservado al holdout final."
        )

    print("=== FASE 16D.8 — ENTRENAMIENTO COMPLETO V4 11-12 ===")
    print(f"Modelo base: {args.base_model}")
    print(f"SHA-256 base: {hash_archivo(args.base_model)}")
    print(f"Timesteps base esperados: {TIMESTEPS_BASE_EXTENSION_V4}")
    print(
        "Timesteps adicionales planificados: "
        f"{TIMESTEPS_COMPLETOS_ADICIONALES_V4}"
    )
    print(
        "Máximo acumulado nominal: "
        f"{TIMESTEPS_ACUMULADOS_MAXIMOS_V4}"
    )
    print(
        "Validación externa: 3 clásicos + "
        f"{len(bateria.semillas_sinteticas)} sintéticos nuevos"
    )
    print(
        "Rango selección externa: "
        f"{min(bateria.semillas_sinteticas)}-"
        f"{max(bateria.semillas_sinteticas)}"
    )
    print("Reward, observación y máscara: SIN CAMBIOS")
    print("Promoción y sobrescritura de modelos: DESACTIVADAS")

    if args.preflight_only:
        model = MaskablePPO.load(str(args.base_model))
        validar_espacios_modelo(
            model,
            timestep_esperado=TIMESTEPS_BASE_EXTENSION_V4,
        )
        resumen = evaluar_modelo_externamente_completo_v4(
            model,
            timestep=int(model.num_timesteps),
            bateria=bateria,
            proveedor_clasicos=proveedor_clasicos,
            configuracion_temporal=configuracion_temporal,
        )
        guardar_json(
            args.preflight_output,
            {
                "fase": "16D.8",
                "tipo": "PREFLIGHT_SIN_ENTRENAMIENTO",
                "modelo_base": args.base_model.resolve(),
                "sha256_modelo_base": hash_archivo(args.base_model),
                "holdout_evidence": args.holdout_evidence.resolve(),
                "sha256_holdout_evidence": hash_archivo(
                    args.holdout_evidence
                ),
                "semillas_por_estrato": semillas_por_estrato(bateria),
                "resumen": resumen.como_dict(),
                "modelo_promovido": False,
            },
        )
        _imprimir_resumen_preflight(resumen)
        print(f"Preflight guardado en: {args.preflight_output}")
        print("RESULTADO: PREFLIGHT_16D_8_OK")
        return

    run_dir = PYTHON_ROOT / "rl_artifacts" / args.run_name.strip()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            "El directorio del run ya existe y no está vacío. Para evitar "
            "sobrescrituras, use otro --run-name o respalde el directorio: "
            f"{run_dir}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    etapas = crear_curriculum_entrenamiento_completo_temporal_v4()
    guardar_json(
        run_dir / "temporal_v4_full_11_12_config.json",
        {
            "version_run": "pedemonte-rl-temporal-v4-full-11-12-v1",
            "fase": "16D.8",
            "run_name": args.run_name,
            "seed": args.seed,
            "n_envs": args.n_envs,
            "modelo_base": args.base_model.resolve(),
            "sha256_modelo_base": hash_archivo(args.base_model),
            "config_base": args.base_config.resolve(),
            "sha256_config_base": hash_archivo(args.base_config),
            "seleccion_base": args.base_selection.resolve(),
            "sha256_seleccion_base": hash_archivo(args.base_selection),
            "resumen_base": args.base_summary.resolve(),
            "sha256_resumen_base": hash_archivo(args.base_summary),
            "temporal_config": args.temporal_config.resolve(),
            "sha256_temporal_config": hash_archivo(args.temporal_config),
            "temporal": temporal_dict,
            "holdout_evidence": args.holdout_evidence.resolve(),
            "sha256_holdout_evidence": hash_archivo(
                args.holdout_evidence
            ),
            "veredicto_holdout_16d7": evidencia_holdout["veredicto"],
            "origen_base_external_best": seleccion_base.get(
                "origen_external_best", ""
            ),
            "timestep_base": resumen_base.get("timestep"),
            "timesteps_adicionales_planificados": (
                TIMESTEPS_COMPLETOS_ADICIONALES_V4
            ),
            "timesteps_acumulados_maximos_planificados": (
                TIMESTEPS_ACUMULADOS_MAXIMOS_V4
            ),
            "observacion": 702,
            "acciones": 30,
            "reward_modificado": False,
            "observacion_modificada": False,
            "mascara_temporal_dura": False,
            "curriculum": [asdict(etapa) for etapa in etapas],
            "validacion_externa": {
                "cantidad_por_estrato": (
                    args.validation_cases_per_stratum
                ),
                "seed_inicio": args.validation_seed_start,
                "semillas_sinteticas": bateria.semillas_sinteticas,
                "semillas_por_estrato": semillas_por_estrato(bateria),
                "incluye_clasicos": [
                    "B04_VENTANAS",
                    "B05_VOLCADOR",
                    "B06_SPLIT",
                ],
                "prioridad_seleccion": [
                    "preservar_B04_B05_B06",
                    "preservar_3_8",
                    "preservar_9_10",
                    "maximizar_factibilidad_exactamente_12",
                    "minimizar_pedidos_tardios_exactamente_12",
                    "minimizar_tardanza_exactamente_12",
                    "mejorar_general_11_12",
                    "mejorar_total_11_12",
                    "evitar_costos_extremos",
                    "costo_mediano_vs_greedy",
                ],
            },
            "internal_eval_seed_start": args.internal_eval_seed_start,
            "continuacion_entre_etapas": "EXTERNAL_BEST_FULL_11_12",
            "modelo_historico_sobrescrito": False,
            "modelo_v4_quick_sobrescrito": False,
            "modelo_extension_9_12_sobrescrito": False,
            "modelo_promovido": False,
        },
    )

    print(f"Directorio del run: {run_dir}")
    print(f"Seed de entrenamiento: {args.seed}")
    print(f"Entornos paralelos: {args.n_envs}")

    model: MaskablePPO | None = None
    selected_model_path: Path = args.base_model
    train_env: DummyVecEnv | None = None
    eval_env: DummyVecEnv | None = None
    baseline_guardado = False

    try:
        for indice, etapa in enumerate(etapas, start=1):
            stage_dir = run_dir / etapa.nombre
            checkpoint_dir = stage_dir / "checkpoints"
            internal_best_dir = stage_dir / "internal_best"
            internal_eval_dir = stage_dir / "internal_evaluation"
            for directorio in (
                checkpoint_dir,
                internal_best_dir,
                internal_eval_dir,
            ):
                directorio.mkdir(parents=True, exist_ok=True)

            seed_etapa = args.seed + indice * 1_000_000
            train_env = DummyVecEnv(
                [
                    crear_env_factory(
                        rank=rank,
                        seed_base=seed_etapa,
                        etapa=etapa,
                        configuracion_reward=configuracion_reward,
                        configuracion_temporal=configuracion_temporal,
                    )
                    for rank in range(args.n_envs)
                ]
            )
            inicio_eval = args.internal_eval_seed_start + indice * 100
            semillas_eval = list(range(inicio_eval, inicio_eval + 40))
            eval_env = DummyVecEnv(
                [
                    crear_env_factory(
                        rank=0,
                        seed_base=semillas_eval[0],
                        etapa=etapa,
                        configuracion_reward=configuracion_reward,
                        configuracion_temporal=configuracion_temporal,
                        semillas_fijas=semillas_eval,
                    )
                ]
            )

            model = MaskablePPO.load(
                str(selected_model_path),
                env=train_env,
            )
            validar_espacios_modelo(model)

            guardar_json(
                stage_dir / "stage_config.json",
                {
                    "indice_full": indice,
                    "etapa": asdict(etapa),
                    "modelo_inicio": selected_model_path.resolve(),
                    "sha256_modelo_inicio": hash_archivo(
                        selected_model_path
                    ),
                    "timestep_inicio": int(model.num_timesteps),
                    "semillas_eval_interna": semillas_eval,
                },
            )

            callback_externo = ValidacionCompletaV4Callback(
                eval_freq=max(etapa.eval_freq // args.n_envs, 1),
                stage_dir=stage_dir,
                bateria=bateria,
                proveedor_clasicos=proveedor_clasicos,
                configuracion_temporal=configuracion_temporal,
                verbose=1,
            )
            callbacks = CallbackList(
                [
                    CheckpointCallback(
                        save_freq=max(
                            etapa.checkpoint_freq // args.n_envs,
                            1,
                        ),
                        save_path=str(checkpoint_dir),
                        name_prefix=(
                            "maskable_ppo_temporal_v4_full_11_12"
                        ),
                        verbose=1,
                    ),
                    MaskableEvalCallback(
                        eval_env,
                        best_model_save_path=str(internal_best_dir),
                        log_path=str(internal_eval_dir),
                        eval_freq=max(etapa.eval_freq // args.n_envs, 1),
                        n_eval_episodes=30,
                        deterministic=True,
                        render=False,
                        verbose=1,
                    ),
                    callback_externo,
                ]
            )

            print("")
            print(f"=== {etapa.nombre} ===")
            print(
                "Banda actual: "
                f"{etapa.min_pedidos_finales}-"
                f"{etapa.max_pedidos_finales}"
            )
            print(f"Timesteps nominales: {etapa.timesteps}")
            print(
                "Mezcla: actual="
                f"{etapa.probabilidad_banda_actual:.0%}, "
                "replay 3-8="
                f"{etapa.probabilidad_replay_3_8:.0%}, "
                "replay 9-10="
                f"{etapa.probabilidad_replay_9_10:.0%}, "
                "general 11-12="
                f"{etapa.probabilidad_replay_general_11_12:.0%}, "
                "exactos 12="
                f"{etapa.probabilidad_replay_exactos_12:.0%}"
            )

            resumen_entrada = callback_externo.evaluar_modelo_y_guardar(
                model,
                int(model.num_timesteps),
            )
            if not baseline_guardado:
                guardar_json(
                    run_dir / "baseline_external_summary.json",
                    resumen_entrada.como_dict(),
                )
                baseline_guardado = True

            model.learn(
                total_timesteps=etapa.timesteps,
                callback=callbacks,
                reset_num_timesteps=False,
            )
            model.save(str(stage_dir / "final_model"))
            callback_externo.evaluar_modelo_y_guardar(
                model,
                int(model.num_timesteps),
            )

            selected_model_path = callback_externo.best_model_path
            if not selected_model_path.is_file():
                raise RuntimeError(
                    "La etapa no generó external_best/best_model.zip."
                )
            shutil.copy2(
                selected_model_path,
                stage_dir / "external_selected_model.zip",
            )
            print(
                "Modelo elegido para la siguiente etapa: "
                f"{selected_model_path}"
            )

            train_env.close()
            eval_env.close()
            train_env = None
            eval_env = None

        if not selected_model_path.is_file():
            raise RuntimeError("No se seleccionó un modelo final.")

        ruta_final = run_dir / "final_model.zip"
        shutil.copy2(selected_model_path, ruta_final)
        resumen_final_path = selected_model_path.parent / "best_summary.json"
        if not resumen_final_path.is_file():
            raise RuntimeError("Falta el resumen del modelo seleccionado.")
        shutil.copy2(
            resumen_final_path,
            run_dir / "final_external_summary.json",
        )
        resumen_final = json.loads(
            resumen_final_path.read_text(encoding="utf-8")
        )

        guardar_json(
            run_dir / "final_model_selection.json",
            {
                "modelo_final": ruta_final.resolve(),
                "sha256_modelo_final": hash_archivo(ruta_final),
                "origen_external_best": selected_model_path.resolve(),
                "criterio": (
                    "VALIDACION_EXTERNA_FULL_11_12_"
                    "LEXICOGRAFICA_V4"
                ),
                "modelo_base": args.base_model.resolve(),
                "timestep_base": TIMESTEPS_BASE_EXTENSION_V4,
                "timestep_final_seleccionado": resumen_final.get(
                    "timestep"
                ),
                "timestep_acumulado_maximo_planificado": (
                    TIMESTEPS_ACUMULADOS_MAXIMOS_V4
                ),
                "modelo_promovido": False,
                "modelo_historico_sobrescrito": False,
                "modelo_v4_quick_sobrescrito": False,
                "modelo_extension_9_12_sobrescrito": False,
            },
        )

        print("")
        print("Entrenamiento completo temporal v4 finalizado.")
        print(f"Modelo final seleccionado: {ruta_final}")
        print(
            "Timestep del checkpoint seleccionado: "
            f"{resumen_final.get('timestep')}"
        )
        print("Modelo promovido: NO")
        print("RESULTADO: ENTRENAMIENTO_16D_8_OK")

    except KeyboardInterrupt:
        if model is not None:
            model.save(str(run_dir / "interrupted_model"))
        print("Entrenamiento interrumpido; modelo interrumpido guardado.")
        print("RESULTADO: ENTRENAMIENTO_16D_8_INTERRUMPIDO")
        raise
    finally:
        if train_env is not None:
            train_env.close()
        if eval_env is not None:
            eval_env.close()


if __name__ == "__main__":
    main()
