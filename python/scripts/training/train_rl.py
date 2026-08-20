from __future__ import annotations

import argparse
import hashlib
import json
import sys

from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

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
from stable_baselines3.common.monitor import Monitor

# pyrefly: ignore [missing-import]
from stable_baselines3.common.vec_env import DummyVecEnv


PYTHON_ROOT = Path(__file__).resolve().parents[2]

if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))


from planner.data.real_demand import (  # noqa: E402
    CatalogoDemandaReal,
)
from planner.rl.instance_generator import (  # noqa: E402
    ConfiguracionGeneradorInstancias,
    GeneradorInstanciasRL,
    ModoDemandaGeografica,
)
from planner.rl.rl_training_env import (  # noqa: E402
    PedemonteTrainingEnv,
)


INFO_KEYWORDS = (
    "costo_estimado",
    "cantidad_viajes",
    "seed_instancia",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Entrena MaskablePPO sobre el entorno de "
            "planificación Pedemonte."
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

    parser.add_argument(
        "--demand-mode",
        choices=[
            ModoDemandaGeografica.SINTETICA.value,
            ModoDemandaGeografica.REAL.value,
        ],
        default=ModoDemandaGeografica.SINTETICA.value,
        help=(
            "Distribución geográfica utilizada para "
            "generar los pedidos."
        ),
    )

    parser.add_argument(
        "--demand-dataset",
        type=str,
        default="",
        help=(
            "CSV procesado de demanda real. Si se omite "
            "en modo REAL se utiliza la ruta predeterminada "
            "del proyecto."
        ),
    )

    parser.add_argument(
        "--large-order-probability",
        type=float,
        default=None,
        help=(
            "Probabilidad de generar pedidos mayores que la "
            "capacidad. Por defecto es 0.10 en modo SINTETICA "
            "y 0.0 en modo REAL."
        ),
    )

    args = parser.parse_args()

    if args.timesteps <= 0:
        parser.error("--timesteps debe ser > 0.")

    if args.n_envs <= 0:
        parser.error("--n-envs debe ser > 0.")

    if args.eval_freq <= 0:
        parser.error("--eval-freq debe ser > 0.")

    if args.checkpoint_freq <= 0:
        parser.error("--checkpoint-freq debe ser > 0.")

    if not args.run_name.strip():
        parser.error("--run-name no puede estar vacío.")

    if (
        args.large_order_probability is not None
        and not 0.0 <= args.large_order_probability <= 1.0
    ):
        parser.error(
            "--large-order-probability debe estar entre 0 y 1."
        )

    return args


def resolver_probabilidad_pedido_grande(
    args: argparse.Namespace,
    modo: ModoDemandaGeografica,
) -> float:
    if args.large_order_probability is not None:
        return float(args.large_order_probability)

    if modo == ModoDemandaGeografica.REAL:
        return 0.0

    return 0.10


def crear_configuracion_generador(
    args: argparse.Namespace,
) -> ConfiguracionGeneradorInstancias:
    modo = ModoDemandaGeografica(args.demand_mode)

    return ConfiguracionGeneradorInstancias(
        min_pedidos_finales=4,
        max_pedidos_finales=8,
        probabilidad_pedido_mayor_capacidad=(
            resolver_probabilidad_pedido_grande(
                args=args,
                modo=modo,
            )
        ),
        modo_demanda_geografica=modo,
        ruta_demanda_real=args.demand_dataset.strip(),
    )


def preparar_catalogo_compartido(
    configuracion: ConfiguracionGeneradorInstancias,
) -> tuple[
    CatalogoDemandaReal | None,
    Path | None,
]:
    if (
        configuracion.modo_demanda_geografica
        == ModoDemandaGeografica.SINTETICA
    ):
        return None, None

    generador_temporal = GeneradorInstanciasRL(
        configuracion=configuracion
    )

    catalogo = generador_temporal.catalogo_demanda_real

    if catalogo is None:
        raise RuntimeError(
            "No se inicializó el catálogo de demanda real."
        )

    return (
        catalogo,
        generador_temporal.ruta_demanda_real_resuelta,
    )


def crear_env_factory(
    rank: int,
    seed_base: int,
    configuracion_generador: ConfiguracionGeneradorInstancias,
    catalogo_demanda_real: CatalogoDemandaReal | None,
) -> Callable[[], gym.Env]:
    def _crear() -> gym.Env:
        generador = GeneradorInstanciasRL(
            configuracion=configuracion_generador,
            catalogo_demanda_real=catalogo_demanda_real,
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
            info_keywords=INFO_KEYWORDS,
        )

    return _crear


def crear_eval_env(
    configuracion_generador: ConfiguracionGeneradorInstancias,
    catalogo_demanda_real: CatalogoDemandaReal | None,
) -> gym.Env:
    semillas_evaluacion = list(
        range(
            150_000,
            150_020,
        )
    )

    generador = GeneradorInstanciasRL(
        configuracion=configuracion_generador,
        catalogo_demanda_real=catalogo_demanda_real,
    )

    env = PedemonteTrainingEnv(
        generador=generador,
        semillas_fijas=semillas_evaluacion,
        max_pedidos=30,
        escala_reward=100.0,
    )

    return Monitor(
        env,
        info_keywords=INFO_KEYWORDS,
    )


def calcular_sha256(ruta: Path | None) -> str:
    if ruta is None:
        return ""

    digest = hashlib.sha256()

    with ruta.open("rb") as archivo:
        for bloque in iter(
            lambda: archivo.read(1024 * 1024),
            b"",
        ):
            digest.update(bloque)

    return digest.hexdigest()


def serializar_configuracion_generador(
    configuracion: ConfiguracionGeneradorInstancias,
) -> dict[str, Any]:
    datos: dict[str, Any] = {}

    for clave, valor in asdict(configuracion).items():
        datos[str(clave)] = valor

    datos["modo_demanda_geografica"] = (
        configuracion.modo_demanda_geografica.value
    )

    return datos


def main() -> None:
    args = parse_args()

    run_dir = (
        PYTHON_ROOT
        / "rl_artifacts"
        / args.run_name
    )

    checkpoints_dir = run_dir / "checkpoints"
    best_dir = run_dir / "best"
    eval_dir = run_dir / "evaluation"

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
        crear_configuracion_generador(args)
    )

    (
        catalogo_demanda_real,
        ruta_demanda_real,
    ) = preparar_catalogo_compartido(
        configuracion_generador
    )

    train_env = DummyVecEnv(
        [
            crear_env_factory(
                rank=rank,
                seed_base=args.seed,
                configuracion_generador=(
                    configuracion_generador
                ),
                catalogo_demanda_real=(
                    catalogo_demanda_real
                ),
            )
            for rank in range(args.n_envs)
        ]
    )

    eval_env = DummyVecEnv(
        [
            lambda: crear_eval_env(
                configuracion_generador=(
                    configuracion_generador
                ),
                catalogo_demanda_real=(
                    catalogo_demanda_real
                ),
            )
        ]
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(
            args.checkpoint_freq
            // args.n_envs,
            1,
        ),
        save_path=str(checkpoints_dir),
        name_prefix="maskable_ppo",
    )

    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=str(best_dir),
        log_path=str(eval_dir),
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

    configuracion_salida: dict[str, Any] = {
        "argumentos": vars(args),
        "generador": serializar_configuracion_generador(
            configuracion_generador
        ),
        "demanda_geografica": {
            "modo": (
                configuracion_generador
                .modo_demanda_geografica
                .value
            ),
            "ruta_dataset_resuelta": (
                str(ruta_demanda_real)
                if ruta_demanda_real is not None
                else ""
            ),
            "sha256_dataset": calcular_sha256(
                ruta_demanda_real
            ),
            "registros_aptos": (
                len(catalogo_demanda_real)
                if catalogo_demanda_real is not None
                else 0
            ),
            "direcciones_fuente_unicas": (
                catalogo_demanda_real
                .cantidad_direcciones_fuente_unicas()
                if catalogo_demanda_real is not None
                else 0
            ),
        },
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
    print("=== ENTRENAMIENTO MASKABLE PPO ===")
    print(f"Timesteps: {args.timesteps}")
    print(f"Entornos paralelos: {args.n_envs}")
    print(f"Seed: {args.seed}")
    print(
        "Demanda geográfica: "
        f"{configuracion_generador.modo_demanda_geografica.value}"
    )

    if ruta_demanda_real is not None:
        print(f"Dataset: {ruta_demanda_real}")
        print(
            "Registros aptos: "
            f"{len(catalogo_demanda_real or ())}"
        )

    print(f"Directorio: {run_dir}")

    final_model_path = run_dir / "final_model"

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            reset_num_timesteps=True,
        )

        model.save(str(final_model_path))
    except KeyboardInterrupt:
        interrupted_path = run_dir / "interrupted_model"
        model.save(str(interrupted_path))
        print("")
        print(
            "Entrenamiento interrumpido. Modelo guardado en: "
            f"{interrupted_path}.zip"
        )
        raise
    finally:
        train_env.close()
        eval_env.close()

    print("")
    print("Entrenamiento finalizado.")
    print(f"Modelo final: {final_model_path}.zip")
    print(f"Mejor modelo: {best_dir / 'best_model.zip'}")


if __name__ == "__main__":
    main()