from __future__ import annotations

import argparse
import json
import sys

from dataclasses import asdict
from pathlib import Path
from typing import Callable

import gymnasium as gym

from sb3_contrib import MaskablePPO

from sb3_contrib.common.maskable.callbacks import (
    MaskableEvalCallback,
)

from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
)

from stable_baselines3.common.monitor import (
    Monitor,
)

from stable_baselines3.common.vec_env import (
    DummyVecEnv,
)


# ============================================================
# RAÍZ PYTHON DEL PROYECTO
# ============================================================
#
# Ubicación:
#
# python/
# └── scripts/
#     └── training/
#         └── train_rl_curriculum.py
#
# parents[0] = training
# parents[1] = scripts
# parents[2] = python

PYTHON_ROOT = Path(
    __file__
).resolve().parents[2]

PLANNER_DIR = (
    PYTHON_ROOT
    / "planner"
)

if not PLANNER_DIR.is_dir():
    raise RuntimeError(
        "No se encontró el paquete planner. "
        f"Raíz calculada: {PYTHON_ROOT}"
    )

python_root_texto = str(
    PYTHON_ROOT
)

if python_root_texto not in sys.path:
    sys.path.insert(
        0,
        python_root_texto,
    )


# ============================================================
# IMPORTS DEL PROYECTO
# ============================================================

from planner.rl.instance_generator import (  # noqa: E402
    ConfiguracionGeneradorInstancias,
    GeneradorInstanciasRL,
)

from planner.rl.rl_curriculum import (  # noqa: E402
    EtapaCurriculumRL,
    crear_curriculum_fase9c,
    crear_curriculum_rapido_fase9c,
)

from planner.rl.rl_reward import (  # noqa: E402
    ConfiguracionRewardRL,
    ModoRewardRL,
)

from planner.rl.rl_training_env import (  # noqa: E402
    PedemonteTrainingEnv,
)


INFO_KEYWORDS = (
    "costo_estimado",
    "cantidad_viajes",
    "seed_instancia",
    "costo_greedy_referencia",
    "gap_relativo_greedy",
)


def parse_args(
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Entrena MaskablePPO mediante "
            "un currículo de instancias Pedemonte."
        )
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default="phase9c_curriculum",
        help=(
            "Nombre del directorio de resultados "
            "dentro de rl_artifacts."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=97_000,
        help=(
            "Seed base del entrenamiento."
        ),
    )

    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help=(
            "Cantidad de entornos vectorizados."
        ),
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Ejecuta el currículo reducido "
            "utilizado como prueba técnica."
        ),
    )

    args = parser.parse_args()

    if not args.run_name.strip():
        parser.error(
            "--run-name no puede estar vacío."
        )

    if args.n_envs <= 0:
        parser.error(
            "--n-envs debe ser > 0."
        )

    return args


def crear_train_env_factory(
    rank: int,
    seed_base: int,
    configuracion_generador:
        ConfiguracionGeneradorInstancias,
    configuracion_reward:
        ConfiguracionRewardRL,
) -> Callable[
    [],
    gym.Env,
]:
    def _crear(
    ) -> gym.Env:
        env = PedemonteTrainingEnv(
            generador=(
                GeneradorInstanciasRL(
                    configuracion_generador
                )
            ),

            seed_base=(
                seed_base
                + rank * 100_000
            ),

            max_pedidos=30,

            escala_reward=100.0,

            configuracion_reward=(
                configuracion_reward
            ),
        )

        return Monitor(
            env,

            info_keywords=(
                INFO_KEYWORDS
            ),
        )

    return _crear


def crear_eval_env_factory(
    semillas: list[int],
    configuracion_generador:
        ConfiguracionGeneradorInstancias,
    configuracion_reward:
        ConfiguracionRewardRL,
) -> Callable[
    [],
    gym.Env,
]:
    def _crear(
    ) -> gym.Env:
        env = PedemonteTrainingEnv(
            generador=(
                GeneradorInstanciasRL(
                    configuracion_generador
                )
            ),

            semillas_fijas=(
                semillas
            ),

            max_pedidos=30,

            escala_reward=100.0,

            configuracion_reward=(
                configuracion_reward
            ),
        )

        return Monitor(
            env,

            info_keywords=(
                INFO_KEYWORDS
            ),
        )

    return _crear


def crear_modelo(
    env: DummyVecEnv,
    seed: int,
) -> MaskablePPO:
    return MaskablePPO(
        policy="MlpPolicy",

        env=env,

        learning_rate=3e-4,

        n_steps=256,

        batch_size=256,

        n_epochs=10,

        gamma=1.0,

        gae_lambda=0.95,

        clip_range=0.2,

        ent_coef=0.01,

        policy_kwargs={
            "net_arch": {
                "pi": [
                    256,
                    256,
                ],

                "vf": [
                    256,
                    256,
                ],
            }
        },

        seed=seed,

        verbose=1,

        device="auto",
    )


def guardar_json(
    ruta: Path,
    datos: dict,
) -> None:
    with ruta.open(
        "w",
        encoding="utf-8",
    ) as archivo:
        json.dump(
            datos,
            archivo,
            indent=2,
            ensure_ascii=False,
        )


def crear_directorios_etapa(
    run_dir: Path,
    etapa: EtapaCurriculumRL,
) -> tuple[
    Path,
    Path,
    Path,
    Path,
]:
    stage_dir = (
        run_dir
        / etapa.nombre
    )

    checkpoint_dir = (
        stage_dir
        / "checkpoints"
    )

    best_dir = (
        stage_dir
        / "best"
    )

    evaluation_dir = (
        stage_dir
        / "evaluation"
    )

    for directorio in (
        stage_dir,
        checkpoint_dir,
        best_dir,
        evaluation_dir,
    ):
        directorio.mkdir(
            parents=True,
            exist_ok=True,
        )

    return (
        stage_dir,
        checkpoint_dir,
        best_dir,
        evaluation_dir,
    )


def main(
) -> None:
    args = parse_args()

    run_dir = (
        PYTHON_ROOT
        / "rl_artifacts"
        / args.run_name
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.quick:
        etapas = (
            crear_curriculum_rapido_fase9c()
        )

    else:
        etapas = (
            crear_curriculum_fase9c()
        )

    configuracion_reward = (
        ConfiguracionRewardRL(
            modo=(
                ModoRewardRL
                .VENTAJA_GREEDY_RELATIVA
            ),

            denominador_relativo_minimo=1.0,
        )
    )

    configuracion_salida = {
        "run_name": (
            args.run_name
        ),

        "seed": (
            args.seed
        ),

        "n_envs": (
            args.n_envs
        ),

        "quick": (
            args.quick
        ),

        "reward": {
            "modo": (
                configuracion_reward
                .modo
                .value
            ),

            "escala_absoluta": (
                configuracion_reward
                .escala_absoluta
            ),

            "denominador_relativo_minimo": (
                configuracion_reward
                .denominador_relativo_minimo
            ),
        },

        "algoritmo": {
            "policy": "MlpPolicy",

            "learning_rate": 3e-4,

            "n_steps": 256,

            "batch_size": 256,

            "n_epochs": 10,

            "gamma": 1.0,

            "gae_lambda": 0.95,

            "clip_range": 0.2,

            "ent_coef": 0.01,

            "net_arch": {
                "pi": [
                    256,
                    256,
                ],

                "vf": [
                    256,
                    256,
                ],
            },
        },

        "etapas": [
            asdict(
                etapa
            )

            for etapa in etapas
        ],
    }

    guardar_json(
        run_dir
        / "curriculum_config.json",

        configuracion_salida,
    )

    model: MaskablePPO | None = None

    train_env: DummyVecEnv | None = None

    eval_env: DummyVecEnv | None = None

    try:
        for indice, etapa in enumerate(
            etapas,
            start=1,
        ):
            print("")
            print(
                "================================"
            )

            print(
                f"=== {etapa.nombre} ==="
            )

            print(
                "================================"
            )

            print(
                "Pedidos finales: "
                f"{etapa.min_pedidos_finales}"
                "-"
                f"{etapa.max_pedidos_finales}"
            )

            print(
                "Timesteps de etapa: "
                f"{etapa.timesteps}"
            )

            (
                stage_dir,
                checkpoint_dir,
                best_dir,
                evaluation_dir,
            ) = crear_directorios_etapa(
                run_dir,
                etapa,
            )

            configuracion_generador = (
                ConfiguracionGeneradorInstancias(
                    min_pedidos_finales=(
                        etapa
                        .min_pedidos_finales
                    ),

                    max_pedidos_finales=(
                        etapa
                        .max_pedidos_finales
                    ),
                )
            )

            guardar_json(
                stage_dir
                / "stage_config.json",

                {
                    "indice": indice,

                    "etapa": asdict(
                        etapa
                    ),

                    "generador": asdict(
                        configuracion_generador
                    ),
                },
            )

            seed_etapa = (
                args.seed
                + indice * 1_000_000
            )

            train_env = DummyVecEnv(
                [
                    crear_train_env_factory(
                        rank=rank,

                        seed_base=(
                            seed_etapa
                        ),

                        configuracion_generador=(
                            configuracion_generador
                        ),

                        configuracion_reward=(
                            configuracion_reward
                        ),
                    )

                    for rank in range(
                        args.n_envs
                    )
                ]
            )

            semillas_evaluacion = list(
                range(
                    180_000
                    + indice * 1_000,

                    180_030
                    + indice * 1_000,
                )
            )

            eval_env = DummyVecEnv(
                [
                    crear_eval_env_factory(
                        semillas=(
                            semillas_evaluacion
                        ),

                        configuracion_generador=(
                            configuracion_generador
                        ),

                        configuracion_reward=(
                            configuracion_reward
                        ),
                    )
                ]
            )

            if model is None:
                model = crear_modelo(
                    train_env,
                    args.seed,
                )

            else:
                model.set_env(
                    train_env
                )

            checkpoint_callback = (
                CheckpointCallback(
                    save_freq=max(
                        etapa.checkpoint_freq
                        // args.n_envs,

                        1,
                    ),

                    save_path=str(
                        checkpoint_dir
                    ),

                    name_prefix=(
                        "maskable_ppo"
                    ),

                    verbose=1,
                )
            )

            eval_callback = (
                MaskableEvalCallback(
                    eval_env,

                    best_model_save_path=str(
                        best_dir
                    ),

                    log_path=str(
                        evaluation_dir
                    ),

                    eval_freq=max(
                        etapa.eval_freq
                        // args.n_envs,

                        1,
                    ),

                    n_eval_episodes=30,

                    deterministic=True,

                    render=False,

                    verbose=1,
                )
            )

            callbacks = CallbackList(
                [
                    checkpoint_callback,
                    eval_callback,
                ]
            )

            model.learn(
                total_timesteps=(
                    etapa.timesteps
                ),

                callback=callbacks,

                reset_num_timesteps=False,
            )

            model.save(
                str(
                    stage_dir
                    / "final_model"
                )
            )

            train_env.close()

            eval_env.close()

            train_env = None

            eval_env = None

            print("")
            print(
                "Etapa finalizada: "
                f"{etapa.nombre}"
            )

            print(
                "Modelo final de etapa: "
                f"{stage_dir / 'final_model.zip'}"
            )

            print(
                "Mejor modelo de etapa: "
                f"{best_dir / 'best_model.zip'}"
            )

        if model is None:
            raise RuntimeError(
                "No se creó ningún modelo."
            )

        model.save(
            str(
                run_dir
                / "final_model"
            )
        )

        print("")
        print(
            "Currículo finalizado."
        )

        print(
            "Modelo final acumulado: "
            f"{run_dir / 'final_model.zip'}"
        )

    except KeyboardInterrupt:
        if model is not None:
            model.save(
                str(
                    run_dir
                    / "interrupted_model"
                )
            )

        print("")
        print(
            "Entrenamiento interrumpido."
        )

        print(
            "Modelo guardado en: "
            f"{run_dir / 'interrupted_model.zip'}"
        )

        raise

    finally:
        if train_env is not None:
            train_env.close()

        if eval_env is not None:
            eval_env.close()


if __name__ == "__main__":
    main()