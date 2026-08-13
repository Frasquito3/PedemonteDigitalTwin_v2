import argparse
import json

from dataclasses import asdict
from pathlib import Path
from typing import Callable

# pyrefly: ignore [missing-import]
import gymnasium as gym

# pyrefly: ignore [missing-import]
from sb3_contrib import MaskablePPO

# pyrefly: ignore [missing-import]
from sb3_contrib.common.maskable.callbacks import (
    MaskableEvalCallback,
)

# pyrefly: ignore [missing-import]
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
)

# pyrefly: ignore [missing-import]
from stable_baselines3.common.monitor import (
    Monitor,
)

# pyrefly: ignore [missing-import]
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
)

from planner.rl.instance_generator import (
    ConfiguracionGeneradorInstancias,
    GeneradorInstanciasRL,
)

from planner.rl.rl_training_env import (
    PedemonteTrainingEnv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Entrena MaskablePPO sobre el "
            "entorno de planificación Pedemonte."
        )
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=20_000,
    )

    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=95_000,
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default="phase9b_smoke",
    )

    parser.add_argument(
        "--eval-freq",
        type=int,
        default=5_000,
    )

    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=10_000,
    )

    args = parser.parse_args()

    if args.timesteps <= 0:
        parser.error(
            "--timesteps debe ser > 0."
        )

    if args.n_envs <= 0:
        parser.error(
            "--n-envs debe ser > 0."
        )

    return args


def crear_env_factory(
    rank: int,
    seed_base: int,
    configuracion_generador:
        ConfiguracionGeneradorInstancias,
) -> Callable[
    [],
    gym.Env,
]:
    def _crear() -> gym.Env:
        generador = GeneradorInstanciasRL(
            configuracion_generador
        )

        env = PedemonteTrainingEnv(
            generador=generador,

            seed_base=(
                seed_base
                + rank * 100_000
            ),

            max_pedidos=30,

            escala_reward=100.0,
        )

        return Monitor(
            env,

            info_keywords=(
                "costo_estimado",
                "cantidad_viajes",
                "seed_instancia",
            ),
        )

    return _crear


def crear_eval_env(
    configuracion_generador:
        ConfiguracionGeneradorInstancias,
) -> gym.Env:
    semillas_evaluacion = list(
        range(
            150_000,
            150_020,
        )
    )

    generador = GeneradorInstanciasRL(
        configuracion_generador
    )

    env = PedemonteTrainingEnv(
        generador=generador,

        semillas_fijas=(
            semillas_evaluacion
        ),

        max_pedidos=30,

        escala_reward=100.0,
    )

    return Monitor(
        env,

        info_keywords=(
            "costo_estimado",
            "cantidad_viajes",
            "seed_instancia",
        ),
    )


def main() -> None:
    args = parse_args()

    base_dir = Path(
        __file__
    ).resolve().parent

    run_dir = (
        base_dir
        / "rl_artifacts"
        / args.run_name
    )

    checkpoints_dir = (
        run_dir
        / "checkpoints"
    )

    best_dir = (
        run_dir
        / "best"
    )

    eval_dir = (
        run_dir
        / "evaluation"
    )

    for directorio in (
        run_dir,
        checkpoints_dir,
        best_dir,
        eval_dir,
    ):
        directorio.mkdir(
            parents=True,
            exist_ok=True,
        )

    configuracion_generador = (
        ConfiguracionGeneradorInstancias(
            min_pedidos_finales=4,
            max_pedidos_finales=8,
        )
    )

    train_env = DummyVecEnv(
        [
            crear_env_factory(
                rank=rank,

                seed_base=args.seed,

                configuracion_generador=(
                    configuracion_generador
                ),
            )

            for rank in range(
                args.n_envs
            )
        ]
    )

    eval_env = DummyVecEnv(
        [
            lambda: crear_eval_env(
                configuracion_generador
            )
        ]
    )

    checkpoint_callback = (
        CheckpointCallback(
            save_freq=max(
                args.checkpoint_freq
                // args.n_envs,

                1,
            ),

            save_path=str(
                checkpoints_dir
            ),

            name_prefix="maskable_ppo",
        )
    )

    eval_callback = (
        MaskableEvalCallback(
            eval_env,

            best_model_save_path=str(
                best_dir
            ),

            log_path=str(
                eval_dir
            ),

            eval_freq=max(
                args.eval_freq
                // args.n_envs,

                1,
            ),

            n_eval_episodes=20,

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

    model = MaskablePPO(
        policy="MlpPolicy",

        env=train_env,

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
                "pi": [256, 256],
                "vf": [256, 256],
            }
        },

        seed=args.seed,

        verbose=1,

        device="auto",
    )

    configuracion_salida = {
        "argumentos": vars(args),

        "generador": asdict(
            configuracion_generador
        ),

        "algoritmo": {
            "learning_rate": 3e-4,
            "n_steps": 256,
            "batch_size": 256,
            "n_epochs": 10,
            "gamma": 1.0,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "net_arch": {
                "pi": [256, 256],
                "vf": [256, 256],
            },
        },
    }

    with (
        run_dir
        / "training_config.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as archivo:
        json.dump(
            configuracion_salida,
            archivo,
            indent=2,
            ensure_ascii=False,
        )

    print("")
    print(
        "=== ENTRENAMIENTO MASKABLE PPO ==="
    )

    print(
        f"Timesteps: {args.timesteps}"
    )

    print(
        f"Entornos paralelos: {args.n_envs}"
    )

    print(
        f"Seed: {args.seed}"
    )

    print(
        f"Directorio: {run_dir}"
    )

    model.learn(
        total_timesteps=(
            args.timesteps
        ),

        callback=callbacks,

        reset_num_timesteps=True,
    )

    final_model_path = (
        run_dir
        / "final_model"
    )

    model.save(
        str(
            final_model_path
        )
    )

    train_env.close()

    eval_env.close()

    print("")
    print(
        "Entrenamiento finalizado."
    )

    print(
        "Modelo final: "
        f"{final_model_path}.zip"
    )

    print(
        "Mejor modelo: "
        f"{best_dir / 'best_model.zip'}"
    )


if __name__ == "__main__":
    main()