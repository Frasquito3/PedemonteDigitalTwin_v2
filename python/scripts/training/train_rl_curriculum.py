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
PLANNER_DIR = PYTHON_ROOT / "planner"

if not PLANNER_DIR.is_dir():
    raise RuntimeError(
        "No se encontró el paquete planner. "
        f"Raíz calculada: {PYTHON_ROOT}"
    )

if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))


from planner.data.real_demand import (  # noqa: E402
    CatalogoDemandaReal,
    ParticionDemandaReal,
    SEED_DIVISION_DEMANDA_REAL_V1,
)
from planner.rl.instance_generator import (  # noqa: E402
    ConfiguracionGeneradorInstancias,
    GeneradorInstanciasRL,
    ModoDemandaGeografica,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Entrena MaskablePPO mediante un currículo "
            "de instancias Pedemonte."
        )
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default="phase15_real_demand_curriculum",
        help=(
            "Nombre del directorio de resultados dentro "
            "de python/rl_artifacts."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=153_000,
        help="Seed base del entrenamiento.",
    )

    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Cantidad de entornos vectorizados.",
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Ejecuta el currículo reducido de 14.000 "
            "pasos utilizado como prueba técnica."
        ),
    )

    parser.add_argument(
        "--demand-mode",
        choices=[
            ModoDemandaGeografica.SINTETICA.value,
            ModoDemandaGeografica.REAL.value,
        ],
        default=ModoDemandaGeografica.SINTETICA.value,
        help="Distribución geográfica de los pedidos.",
    )

    parser.add_argument(
        "--demand-dataset",
        type=str,
        default="",
        help=(
            "CSV procesado de demanda real. Si se omite "
            "en modo REAL se usa la ruta predeterminada."
        ),
    )

    parser.add_argument(
        "--large-order-probability",
        type=float,
        default=None,
        help=(
            "Probabilidad de pedidos mayores que la "
            "capacidad. Por defecto: 0.10 en SINTETICA "
            "y 0.0 en REAL."
        ),
    )

    args = parser.parse_args()

    if not args.run_name.strip():
        parser.error("--run-name no puede estar vacío.")

    if args.n_envs <= 0:
        parser.error("--n-envs debe ser > 0.")

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
    min_pedidos: int,
    max_pedidos: int,
    particion_demanda_real: ParticionDemandaReal | None = None,
) -> ConfiguracionGeneradorInstancias:
    modo = ModoDemandaGeografica(args.demand_mode)

    return ConfiguracionGeneradorInstancias(
        min_pedidos_finales=min_pedidos,
        max_pedidos_finales=max_pedidos,
        probabilidad_pedido_mayor_capacidad=(
            resolver_probabilidad_pedido_grande(
                args=args,
                modo=modo,
            )
        ),
        modo_demanda_geografica=modo,
        ruta_demanda_real=args.demand_dataset.strip(),
        particion_demanda_real=(
            particion_demanda_real
            if modo == ModoDemandaGeografica.REAL
            else None
        ),
        seed_division_demanda_real=(
            SEED_DIVISION_DEMANDA_REAL_V1
        ),
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

    datos["particion_demanda_real"] = (
        configuracion.particion_demanda_real.value
        if configuracion.particion_demanda_real
        is not None
        else ""
    )

    return datos


def crear_metadata_demanda(
    configuracion: ConfiguracionGeneradorInstancias,
    catalogo: CatalogoDemandaReal | None,
    ruta_dataset: Path | None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "modo": (
            configuracion.modo_demanda_geografica.value
        ),
        "ruta_dataset_resuelta": (
            str(ruta_dataset)
            if ruta_dataset is not None
            else ""
        ),
        "sha256_dataset": calcular_sha256(
            ruta_dataset
        ),
        "registros_aptos": (
            len(catalogo)
            if catalogo is not None
            else 0
        ),
        "direcciones_fuente_unicas": (
            catalogo.cantidad_direcciones_fuente_unicas()
            if catalogo is not None
            else 0
        ),
    }

    if (
        configuracion.modo_demanda_geografica
        == ModoDemandaGeografica.REAL
    ):
        if catalogo is None:
            raise RuntimeError(
                "El modo REAL requiere un catálogo."
            )

        division = (
            catalogo.dividir_por_direccion_fuente(
                seed=SEED_DIVISION_DEMANDA_REAL_V1
            )
        )
        division.validar_sin_fuga()

        metadata["division"] = {
            "seed": SEED_DIVISION_DEMANDA_REAL_V1,
            "entrenamiento": (
                ParticionDemandaReal.ENTRENAMIENTO.value
            ),
            "seleccion_checkpoint": (
                ParticionDemandaReal.VALIDACION.value
            ),
            "prueba_reservada": (
                ParticionDemandaReal.PRUEBA.value
            ),
            "particiones": division.resumen(),
        }

    return metadata

def crear_train_env_factory(
    rank: int,
    seed_base: int,
    configuracion_generador: ConfiguracionGeneradorInstancias,
    configuracion_reward: ConfiguracionRewardRL,
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
            configuracion_reward=configuracion_reward,
        )

        return Monitor(
            env,
            info_keywords=INFO_KEYWORDS,
        )

    return _crear


def crear_eval_env_factory(
    semillas: list[int],
    configuracion_generador: ConfiguracionGeneradorInstancias,
    configuracion_reward: ConfiguracionRewardRL,
    catalogo_demanda_real: CatalogoDemandaReal | None,
) -> Callable[[], gym.Env]:
    def _crear() -> gym.Env:
        generador = GeneradorInstanciasRL(
            configuracion=configuracion_generador,
            catalogo_demanda_real=catalogo_demanda_real,
        )

        env = PedemonteTrainingEnv(
            generador=generador,
            semillas_fijas=semillas,
            max_pedidos=30,
            escala_reward=100.0,
            configuracion_reward=configuracion_reward,
        )

        return Monitor(
            env,
            info_keywords=INFO_KEYWORDS,
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
                "pi": [256, 256],
                "vf": [256, 256],
            }
        },
        seed=seed,
        verbose=1,
        device="auto",
    )


def guardar_json(
    ruta: Path,
    datos: dict[str, Any],
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
    stage_dir = run_dir / etapa.nombre
    checkpoint_dir = stage_dir / "checkpoints"
    best_dir = stage_dir / "best"
    evaluation_dir = stage_dir / "evaluation"

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


def main() -> None:
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

    etapas = (
        crear_curriculum_rapido_fase9c()
        if args.quick
        else crear_curriculum_fase9c()
    )

    if not etapas:
        raise RuntimeError(
            "El currículo no contiene etapas."
        )

    max_pedidos_curriculum = max(
        etapa.max_pedidos_finales
        for etapa in etapas
    )

    configuracion_catalogo = crear_configuracion_generador(
        args=args,
        min_pedidos=1,
        max_pedidos=max_pedidos_curriculum,
    )

    (
        catalogo_demanda_real,
        ruta_demanda_real,
    ) = preparar_catalogo_compartido(
        configuracion_catalogo
    )

    metadata_demanda = crear_metadata_demanda(
        configuracion=configuracion_catalogo,
        catalogo=catalogo_demanda_real,
        ruta_dataset=ruta_demanda_real,
    )

    configuracion_reward = ConfiguracionRewardRL(
        modo=ModoRewardRL.VENTAJA_GREEDY_RELATIVA,
        denominador_relativo_minimo=1.0,
    )

    configuracion_salida: dict[str, Any] = {
        "run_name": args.run_name,
        "seed": args.seed,
        "n_envs": args.n_envs,
        "quick": args.quick,
        "demanda_geografica": metadata_demanda,
        "probabilidad_pedido_mayor_capacidad": (
            configuracion_catalogo
            .probabilidad_pedido_mayor_capacidad
        ),
        "reward": {
            "modo": configuracion_reward.modo.value,
            "escala_absoluta": (
                configuracion_reward.escala_absoluta
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
                "pi": [256, 256],
                "vf": [256, 256],
            },
        },
        "etapas": [
            asdict(etapa)
            for etapa in etapas
        ],
    }

    guardar_json(
        run_dir / "curriculum_config.json",
        configuracion_salida,
    )

    print("")
    print("=== CURRÍCULO MASKABLE PPO ===")
    print(f"Directorio: {run_dir}")
    print(f"Seed: {args.seed}")
    print(f"Entornos paralelos: {args.n_envs}")
    print(f"Currículo rápido: {args.quick}")
    print(
        "Demanda geográfica: "
        f"{metadata_demanda['modo']}"
    )

    if ruta_demanda_real is not None:
        print(f"Dataset: {ruta_demanda_real}")
        print(
            "Registros aptos: "
            f"{metadata_demanda['registros_aptos']}"
        )
        division_metadata = metadata_demanda.get(
            "division"
        )

        if isinstance(division_metadata, dict):
            print(
                "Partición entrenamiento: "
                f"{ParticionDemandaReal.ENTRENAMIENTO.value}"
            )
            print(
                "Partición selección checkpoint: "
                f"{ParticionDemandaReal.VALIDACION.value}"
            )
            print(
                "TEST reservado: "
                f"{ParticionDemandaReal.PRUEBA.value}"
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
            print("================================")
            print(f"=== {etapa.nombre} ===")
            print("================================")
            print(
                "Pedidos finales: "
                f"{etapa.min_pedidos_finales}-"
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
                run_dir=run_dir,
                etapa=etapa,
            )

            configuracion_generador_train = (
                crear_configuracion_generador(
                    args=args,
                    min_pedidos=(
                        etapa.min_pedidos_finales
                    ),
                    max_pedidos=(
                        etapa.max_pedidos_finales
                    ),
                    particion_demanda_real=(
                        ParticionDemandaReal.ENTRENAMIENTO
                    ),
                )
            )

            configuracion_generador_eval = (
                crear_configuracion_generador(
                    args=args,
                    min_pedidos=(
                        etapa.min_pedidos_finales
                    ),
                    max_pedidos=(
                        etapa.max_pedidos_finales
                    ),
                    particion_demanda_real=(
                        ParticionDemandaReal.VALIDACION
                    ),
                )
            )

            guardar_json(
                stage_dir / "stage_config.json",
                {
                    "indice": indice,
                    "etapa": asdict(etapa),
                    "generador_train": (
                        serializar_configuracion_generador(
                            configuracion_generador_train
                        )
                    ),
                    "generador_validation": (
                        serializar_configuracion_generador(
                            configuracion_generador_eval
                        )
                    ),
                    "demanda_geografica": metadata_demanda,
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
                        seed_base=seed_etapa,
                        configuracion_generador=(
                            configuracion_generador_train
                        ),
                        configuracion_reward=(
                            configuracion_reward
                        ),
                        catalogo_demanda_real=(
                            catalogo_demanda_real
                        ),
                    )
                    for rank in range(args.n_envs)
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
                        semillas=semillas_evaluacion,
                        configuracion_generador=(
                            configuracion_generador_eval
                        ),
                        configuracion_reward=(
                            configuracion_reward
                        ),
                        catalogo_demanda_real=(
                            catalogo_demanda_real
                        ),
                    )
                ]
            )

            if model is None:
                model = crear_modelo(
                    env=train_env,
                    seed=args.seed,
                )
            else:
                model.set_env(train_env)

            checkpoint_callback = CheckpointCallback(
                save_freq=max(
                    etapa.checkpoint_freq
                    // args.n_envs,
                    1,
                ),
                save_path=str(checkpoint_dir),
                name_prefix="maskable_ppo",
                verbose=1,
            )

            eval_callback = MaskableEvalCallback(
                eval_env,
                best_model_save_path=str(best_dir),
                log_path=str(evaluation_dir),
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

            callbacks = CallbackList(
                [
                    checkpoint_callback,
                    eval_callback,
                ]
            )

            model.learn(
                total_timesteps=etapa.timesteps,
                callback=callbacks,
                reset_num_timesteps=False,
            )

            model.save(
                str(stage_dir / "final_model")
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
            str(run_dir / "final_model")
        )

        print("")
        print("Currículo finalizado.")
        print(
            "Modelo final acumulado: "
            f"{run_dir / 'final_model.zip'}"
        )

    except KeyboardInterrupt:
        if model is not None:
            model.save(
                str(run_dir / "interrupted_model")
            )

        print("")
        print("Entrenamiento interrumpido.")
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