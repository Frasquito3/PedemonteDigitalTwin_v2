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
from planner.rl.policy_config import (  # noqa: E402
    ConfiguracionTemporalV4RL,
)
from planner.rl.balanced_policy_curriculum import (  # noqa: E402
    EtapaExtensionTemporalV4RL,
    crear_curriculum_extension_temporal_v4_diagnostico,
)
from planner.rl.balanced_policy_validation import (  # noqa: E402
    BateriaValidacionExtensionV4,
    ResumenValidacionExtensionV4,
    crear_bateria_validacion_extension_v4,
    es_mejor_validacion_extension_v4,
    evaluar_modelo_externamente_extension_v4,
    hash_archivo,
    semillas_por_estrato,
    validar_origen_v4_quick,
)
from planner.rl.policy_training_env import (  # noqa: E402
    PedemonteTemporalV4TrainingEnv,
)
from planner.rl.balanced_policy_generator import (  # noqa: E402
    GeneradorReplayMultibandaTemporalV4RL,
)
from planner.rl.policy_instance_generator import (  # noqa: E402
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


class ValidacionExtensionV4Callback(BaseCallback):
    def __init__(
        self,
        *,
        eval_freq: int,
        stage_dir: Path,
        bateria: BateriaValidacionExtensionV4,
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
        self.best_summary: ResumenValidacionExtensionV4 | None = None

    @property
    def best_model_path(self) -> Path:
        return self.best_dir / "best_model.zip"

    def _guardar_historial(
        self,
        resumen: ResumenValidacionExtensionV4,
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
            "objetivo_9_12_totales": resumen.objetivo_9_12_totales,
            "objetivo_9_12_sin_riesgo": (
                resumen.objetivo_9_12_sin_riesgo
            ),
            "objetivo_9_12_tardanza_total_min": (
                resumen.objetivo_9_12_tardanza_total_min
            ),
            "objetivo_9_12_costos_extremos": (
                resumen.objetivo_9_12_costos_extremos
            ),
            "guard_3_8_totales": resumen.guard_3_8_totales,
            "guard_3_8_sin_riesgo": resumen.guard_3_8_sin_riesgo,
            "guard_3_8_tardanza_total_min": (
                resumen.guard_3_8_tardanza_total_min
            ),
            "guard_3_8_costos_extremos": (
                resumen.guard_3_8_costos_extremos
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
    ) -> ResumenValidacionExtensionV4:
        resumen = evaluar_modelo_externamente_extension_v4(
            model,
            timestep=int(timestep),
            bateria=self.bateria,
            proveedor_clasicos=self.proveedor_clasicos,
            configuracion_temporal=self.configuracion_temporal,
        )
        mejorado = es_mejor_validacion_extension_v4(
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
                "External V4 9-12 "
                f"t={resumen.timestep} | "
                "clásicos tardíos="
                f"{resumen.clasicos_pedidos_tardios} "
                "reg_cost="
                f"{resumen.clasicos_regresiones_costo} | "
                "objetivo 9-12 sin riesgo="
                f"{resumen.objetivo_9_12_sin_riesgo}/"
                f"{resumen.objetivo_9_12_totales} | "
                "guard 3-8 sin riesgo="
                f"{resumen.guard_3_8_sin_riesgo}/"
                f"{resumen.guard_3_8_totales} | {marca}"
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
        PYTHON_ROOT / "rl_artifacts" / "policy_base"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Continúa el mejor modelo temporal v4 quick con un currículo "
            "diagnóstico enfocado en 9-12 pedidos."
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
        default=base_dir / "policy_config.json",
    )
    parser.add_argument(
        "--base-selection",
        type=Path,
        default=base_dir / "final_model_selection.json",
    )
    parser.add_argument(
        "--run-name",
        default="policy_balanced",
    )
    parser.add_argument("--seed", type=int, default=166_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument(
        "--validation-cases-per-stratum",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--validation-seed-start",
        type=int,
        default=271_000,
    )
    args = parser.parse_args()

    if not args.run_name.strip():
        parser.error("--run-name no puede estar vacío.")
    if args.n_envs <= 0:
        parser.error("--n-envs debe ser > 0.")
    if args.validation_cases_per_stratum <= 0:
        parser.error("--validation-cases-per-stratum debe ser > 0.")
    if args.validation_seed_start < 270_000:
        parser.error("--validation-seed-start debe ser >= 270000.")
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


def crear_generador_extension(
    etapa: EtapaExtensionTemporalV4RL,
) -> GeneradorReplayMultibandaTemporalV4RL:
    actual = _generador_temporal(
        _config_generador(
            min_pedidos=etapa.min_pedidos_finales,
            max_pedidos=etapa.max_pedidos_finales,
            probabilidad_ventana=etapa.probabilidad_ventana_especifica,
            probabilidad_volcador=etapa.probabilidad_volcador,
            probabilidad_pedido_grande=etapa.probabilidad_pedido_grande,
            ancho_min=etapa.ancho_ventana_min,
            ancho_max=etapa.ancho_ventana_max,
        ),
        etapa.probabilidad_patron_conflictivo,
    )
    replay_3_8 = _generador_temporal(
        _config_generador(
            min_pedidos=3,
            max_pedidos=8,
            probabilidad_ventana=0.90,
            probabilidad_volcador=0.10,
            probabilidad_pedido_grande=0.03,
            ancho_min=45,
            ancho_max=150,
        ),
        0.70,
    )
    replay_9_10 = _generador_temporal(
        _config_generador(
            min_pedidos=9,
            max_pedidos=10,
            probabilidad_ventana=0.90,
            probabilidad_volcador=0.15,
            probabilidad_pedido_grande=0.05,
            ancho_min=45,
            ancho_max=150,
        ),
        0.75,
    )
    return GeneradorReplayMultibandaTemporalV4RL(
        generador_actual=actual,
        generador_replay_3_8=replay_3_8,
        generador_replay_9_10=replay_9_10,
        probabilidad_replay_3_8=etapa.probabilidad_replay_3_8,
        probabilidad_replay_9_10=etapa.probabilidad_replay_9_10,
    )


def crear_env_factory(
    *,
    rank: int,
    seed_base: int,
    etapa: EtapaExtensionTemporalV4RL,
    configuracion_reward: ConfiguracionRewardRL,
    configuracion_temporal: ConfiguracionTemporalV4RL,
    semillas_fijas: list[int] | None = None,
) -> Callable[[], gym.Env]:
    def _crear() -> gym.Env:
        env = PedemonteTemporalV4TrainingEnv(
            generador=crear_generador_extension(etapa),
            seed_base=seed_base + rank * 100_000,
            semillas_fijas=semillas_fijas,
            max_pedidos=30,
            configuracion_reward=configuracion_reward,
            configuracion_temporal=configuracion_temporal,
        )
        return Monitor(env, info_keywords=INFO_KEYWORDS)

    return _crear


def validar_espacios_modelo(model: Any) -> None:
    forma = getattr(model.observation_space, "shape", None)
    acciones = getattr(model.action_space, "n", None)
    if forma != (702,):
        raise RuntimeError(
            "El modelo base no corresponde al entorno temporal v4. "
            f"Observación encontrada: {forma}; esperada: (702,)."
        )
    if int(acciones) != 30:
        raise RuntimeError(
            "El modelo base no tiene las 30 acciones esperadas."
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


def main() -> None:
    args = parse_args()
    config_base, seleccion_base = validar_origen_v4_quick(
        args.base_config,
        args.base_selection,
    )
    if not args.base_model.is_file():
        raise FileNotFoundError(
            f"No existe el modelo v4 quick base: {args.base_model}"
        )

    run_dir = PYTHON_ROOT / "rl_artifacts" / args.run_name.strip()
    base_dir_resuelto = args.base_model.resolve().parent
    if run_dir.resolve() == base_dir_resuelto:
        raise ValueError(
            "La extensión debe escribirse en un run distinto del v4 quick."
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    etapas = crear_curriculum_extension_temporal_v4_diagnostico()
    temporal_dict = config_base.get("temporal")
    if not isinstance(temporal_dict, dict):
        raise ValueError("La configuración base no contiene temporal.")
    configuracion_temporal = ConfiguracionTemporalV4RL(
        **temporal_dict
    )
    configuracion_reward = ConfiguracionRewardRL(
        modo=ModoRewardRL.VENTAJA_GREEDY_RELATIVA,
        denominador_relativo_minimo=1.0,
    )
    bateria = crear_bateria_validacion_extension_v4(
        cantidad_por_estrato=args.validation_cases_per_stratum,
        seed_inicio=args.validation_seed_start,
    )
    proveedor_clasicos = ProveedorVialCachePersistente(
        PYTHON_ROOT / "data" / "routing" / "cache_vial.csv",
        version_cache_esperada="pedemonte-vial-v1",
        permitir_fallback=False,
    )

    guardar_json(
        run_dir / "balanced_policy_config.json",
        {
            "version_run": (
                "pedemonte-rl-temporal-v4-extension-9-12-v1"
            ),
            "run_name": args.run_name,
            "seed": args.seed,
            "n_envs": args.n_envs,
            "modelo_base": args.base_model.resolve(),
            "sha256_modelo_base": hash_archivo(args.base_model),
            "config_base": args.base_config.resolve(),
            "sha256_config_base": hash_archivo(args.base_config),
            "seleccion_base": args.base_selection.resolve(),
            "sha256_seleccion_base": hash_archivo(
                args.base_selection
            ),
            "origen_base_external_best": seleccion_base.get(
                "origen_external_best", ""
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
                    "preservar_clasicos",
                    "maximizar_factibilidad_9_12",
                    "minimizar_tardanza_9_12",
                    "evitar_costos_extremos_9_12",
                    "preservar_factibilidad_3_8",
                    "minimizar_tardanza_3_8",
                    "costo_mediano_vs_greedy",
                ],
            },
            "continuacion_entre_etapas": "EXTERNAL_BEST_9_12",
            "modelo_historico_sobrescrito": False,
            "modelo_v3_sobrescrito": False,
            "modelo_v4_quick_sobrescrito": False,
            "modelo_promovido": False,
        },
    )

    print("=== EXTENSIÓN RL TEMPORAL V4 — 9 A 12 PEDIDOS ===")
    print(f"Modelo base: {args.base_model}")
    print(f"Directorio: {run_dir}")
    print(f"Seed: {args.seed}")
    print(f"Entornos: {args.n_envs}")
    print(
        "Validación fija: "
        f"3 clásicos + {len(bateria.semillas_sinteticas)} sintéticos"
    )
    print("Reward y observación v4: SIN CAMBIOS")
    print("Modelo base y modelos anteriores: NO SE MODIFICAN")
    print("Promoción: DESACTIVADA")

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
            semillas_eval = list(
                range(
                    270_000 + indice * 100,
                    270_040 + indice * 100,
                )
            )
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
                    "indice_extension": indice,
                    "etapa": asdict(etapa),
                    "modelo_inicio": selected_model_path.resolve(),
                    "semillas_eval_interna": semillas_eval,
                },
            )

            callback_externo = ValidacionExtensionV4Callback(
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
                        name_prefix="rl_policy_balanced",
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
                f"{etapa.probabilidad_replay_9_10:.0%}"
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
            raise RuntimeError("No se seleccionó un modelo de extensión.")

        ruta_final = run_dir / "final_model.zip"
        shutil.copy2(selected_model_path, ruta_final)
        resumen_final_path = selected_model_path.parent / "best_summary.json"
        if resumen_final_path.is_file():
            shutil.copy2(
                resumen_final_path,
                run_dir / "final_external_summary.json",
            )

        guardar_json(
            run_dir / "final_model_selection.json",
            {
                "modelo_final": ruta_final.resolve(),
                "origen_external_best": selected_model_path.resolve(),
                "criterio": (
                    "VALIDACION_EXTERNA_9_12_LEXICOGRAFICA_V4_EXTENSION"
                ),
                "modelo_base": args.base_model.resolve(),
                "modelo_promovido": False,
                "modelo_v4_quick_sobrescrito": False,
            },
        )

        print("")
        print("Extensión temporal v4 9-12 finalizada.")
        print(f"Modelo final seleccionado: {ruta_final}")
        print("Modelo promovido: NO")

    except KeyboardInterrupt:
        if model is not None:
            model.save(str(run_dir / "interrupted_model"))
        print("Entrenamiento interrumpido; checkpoint guardado.")
        raise
    finally:
        if train_env is not None:
            train_env.close()
        if eval_env is not None:
            eval_env.close()


if __name__ == "__main__":
    main()
