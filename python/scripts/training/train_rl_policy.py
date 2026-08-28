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
PLANNER_DIR = PYTHON_ROOT / "planner"

if not PLANNER_DIR.is_dir():
    raise RuntimeError(
        "No se encontró el paquete planner. "
        f"Raíz calculada: {PYTHON_ROOT}"
    )

if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))


from planner.rl.training_data import (  # noqa: E402
    CatalogoDemandaReal,
    ParticionDemandaReal,
    SEED_DIVISION_DEMANDA_REAL,
)
from planner.rl.training_cases import (  # noqa: E402
    crear_casos_benchmark_clasico,
)
from planner.rl.instance_generator import (  # noqa: E402
    ConfiguracionGeneradorInstancias,
    GeneradorInstanciasRL,
    ModoDemandaGeografica,
)
from planner.rl.reward import (  # noqa: E402
    ConfiguracionRewardRL,
    ModoRewardRL,
)
from planner.rl.policy_config import (  # noqa: E402
    ConfiguracionPoliticaRL,
)
from planner.rl.training_curriculum import (  # noqa: E402
    EtapaEntrenamientoRL,
    crear_curriculum_entrenamiento,
    crear_curriculum_entrenamiento_rapido,
)
from planner.rl.training_env import (  # noqa: E402
    EntornoEntrenamientoRL,
)
from planner.rl.training_validation import (  # noqa: E402
    ResumenValidacionExternaRL,
    es_mejor_validacion_externa,
    evaluar_modelo_externamente,
)
from planner.rl.training_instance_generator import (  # noqa: E402
    ConfiguracionGeneradorPoliticaRL,
    GeneradorInstanciasPoliticaRL,
    GeneradorMixtoPoliticaRL,
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
    "reward_terminal",
    "componente_terminal_factibilidad",
    "componente_terminal_costo_acotado",
)


class ValidacionExternaCallback(BaseCallback):
    def __init__(
        self,
        *,
        eval_freq: int,
        stage_dir: Path,
        instancia_b04: Any,
        proveedor_b04: Any,
        instancias_sinteticas: tuple[Any, ...],
        configuracion_temporal: ConfiguracionPoliticaRL,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)

        if eval_freq <= 0:
            raise ValueError("eval_freq debe ser > 0.")

        self.eval_freq = eval_freq
        self.stage_dir = stage_dir
        self.instancia_b04 = instancia_b04
        self.proveedor_b04 = proveedor_b04
        self.instancias_sinteticas = instancias_sinteticas
        self.configuracion_temporal = configuracion_temporal
        self.external_dir = stage_dir / "external_validation"
        self.best_dir = stage_dir / "external_best"
        self.external_dir.mkdir(parents=True, exist_ok=True)
        self.best_dir.mkdir(parents=True, exist_ok=True)
        self.historial_csv = self.external_dir / "history.csv"
        self.best_summary: ResumenValidacionExternaRL | None = None

    @property
    def best_model_path(self) -> Path:
        return self.best_dir / "best_model.zip"

    def _guardar_historial(
        self,
        resumen: ResumenValidacionExternaRL,
        mejorado: bool,
    ) -> None:
        fila = {
            "timestep": resumen.timestep,
            "b04_pedidos_tardios": resumen.b04_pedidos_tardios,
            "b04_tardanza_min": resumen.b04_tardanza_min,
            "b04_costo_estimado": resumen.b04_costo_estimado,
            "sinteticos_totales": resumen.sinteticos_totales,
            "sinteticos_sin_riesgo": resumen.sinteticos_sin_riesgo,
            "tasa_sintetica_sin_riesgo_pct": (
                resumen.tasa_sintetica_sin_riesgo_pct
            ),
            "tardanza_sintetica_total_min": (
                resumen.tardanza_sintetica_total_min
            ),
            "tardanza_sintetica_mediana_min": (
                resumen.tardanza_sintetica_mediana_min
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
    ) -> ResumenValidacionExternaRL:
        resumen = evaluar_modelo_externamente(
            model,
            timestep=int(timestep),
            instancia_b04=self.instancia_b04,
            proveedor_b04=self.proveedor_b04,
            instancias_sinteticas=self.instancias_sinteticas,
            configuracion_temporal=self.configuracion_temporal,
        )
        mejorado = es_mejor_validacion_externa(
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
                "Validación externa "
                f"t={resumen.timestep} | "
                f"B04 tardíos={resumen.b04_pedidos_tardios} "
                f"tardanza={resumen.b04_tardanza_min:.3f} | "
                "sintéticos sin riesgo="
                f"{resumen.sinteticos_sin_riesgo}/"
                f"{resumen.sinteticos_totales} | {marca}"
            )

        return resumen

    def evaluar_y_guardar(self) -> ResumenValidacionExternaRL:
        return self.evaluar_modelo_y_guardar(
            self.model,
            int(self.num_timesteps),
        )

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True
        self.evaluar_y_guardar()
        return True



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Continúa o entrena la política RL actual con máscara temporal "
            "dura, validación externa fija y replay entre etapas."
        )
    )
    parser.add_argument(
        "--run-name",
        default="policy_training",
        help="Nombre estable dentro de python/rl_artifacts.",
    )
    parser.add_argument("--seed", type=int, default=410_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Ejecuta un entrenamiento breve de comprobación.",
    )
    parser.add_argument(
        "--demand-mode",
        choices=[
            ModoDemandaGeografica.SINTETICA.value,
            ModoDemandaGeografica.REAL.value,
        ],
        default=ModoDemandaGeografica.SINTETICA.value,
    )
    parser.add_argument("--demand-dataset", default="")
    parser.add_argument(
        "--initial-model",
        default=str(
            PYTHON_ROOT / "models" / "rl" / "rl_policy.zip"
        ),
        help=(
            "Checkpoint desde el que continúa el entrenamiento. "
            "Por defecto usa la política productiva actual."
        ),
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Crea un modelo nuevo en lugar de continuar el checkpoint actual.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
    )
    args = parser.parse_args()

    if not args.run_name.strip():
        parser.error("--run-name no puede estar vacío.")
    if args.n_envs <= 0:
        parser.error("--n-envs debe ser > 0.")
    if args.learning_rate <= 0.0:
        parser.error("--learning-rate debe ser > 0.")
    if not args.from_scratch and not args.initial_model.strip():
        parser.error("--initial-model no puede estar vacío.")

    return args


def crear_configuracion_generador(
    args: argparse.Namespace,
    etapa: EtapaEntrenamientoRL,
    particion: ParticionDemandaReal | None,
) -> ConfiguracionGeneradorInstancias:
    modo = ModoDemandaGeografica(args.demand_mode)
    return ConfiguracionGeneradorInstancias(
        min_pedidos_finales=etapa.min_pedidos_finales,
        max_pedidos_finales=etapa.max_pedidos_finales,
        probabilidad_volcador=etapa.probabilidad_volcador,
        probabilidad_ventana_especifica=(
            etapa.probabilidad_ventana_especifica
        ),
        probabilidad_pedido_mayor_capacidad=(
            etapa.probabilidad_pedido_grande
        ),
        ancho_ventana_min=etapa.ancho_ventana_min,
        ancho_ventana_max=etapa.ancho_ventana_max,
        modo_demanda_geografica=modo,
        ruta_demanda_real=args.demand_dataset.strip(),
        particion_demanda_real=(
            particion
            if modo == ModoDemandaGeografica.REAL
            else None
        ),
        seed_division_demanda_real=SEED_DIVISION_DEMANDA_REAL,
    )



def crear_configuracion_core(
    args: argparse.Namespace,
    particion: ParticionDemandaReal | None,
) -> ConfiguracionGeneradorInstancias:
    modo = ModoDemandaGeografica(args.demand_mode)
    return ConfiguracionGeneradorInstancias(
        min_pedidos_finales=3,
        max_pedidos_finales=5,
        probabilidad_volcador=0.0,
        probabilidad_ventana_especifica=1.0,
        probabilidad_pedido_mayor_capacidad=0.0,
        ancho_ventana_min=45,
        ancho_ventana_max=90,
        modo_demanda_geografica=modo,
        ruta_demanda_real=args.demand_dataset.strip(),
        particion_demanda_real=(
            particion
            if modo == ModoDemandaGeografica.REAL
            else None
        ),
        seed_division_demanda_real=SEED_DIVISION_DEMANDA_REAL,
    )



def preparar_catalogo_real(
    args: argparse.Namespace,
    etapa: EtapaEntrenamientoRL,
) -> CatalogoDemandaReal | None:
    if args.demand_mode == ModoDemandaGeografica.SINTETICA.value:
        return None

    generador = GeneradorInstanciasRL(
        crear_configuracion_generador(args, etapa, None)
    )
    catalogo = generador.catalogo_demanda_real_completo

    if catalogo is None:
        raise RuntimeError(
            "No se pudo inicializar el catálogo de demanda real."
        )
    return catalogo



def crear_generador_mixto(
    *,
    configuracion_actual: ConfiguracionGeneradorInstancias,
    configuracion_core: ConfiguracionGeneradorInstancias,
    etapa: EtapaEntrenamientoRL,
    catalogo_real: CatalogoDemandaReal | None,
) -> GeneradorMixtoPoliticaRL:
    actual = GeneradorInstanciasPoliticaRL(
        GeneradorInstanciasRL(
            configuracion=configuracion_actual,
            catalogo_demanda_real=catalogo_real,
        ),
        ConfiguracionGeneradorPoliticaRL(
            probabilidad_patron_ventanas_conflictivas=(
                etapa.probabilidad_patron_conflictivo
            )
        ),
    )
    core = GeneradorInstanciasPoliticaRL(
        GeneradorInstanciasRL(
            configuracion=configuracion_core,
            catalogo_demanda_real=catalogo_real,
        ),
        ConfiguracionGeneradorPoliticaRL(
            probabilidad_patron_ventanas_conflictivas=0.95
        ),
    )
    return GeneradorMixtoPoliticaRL(
        generador_actual=actual,
        generador_core=core,
        probabilidad_replay_core=etapa.probabilidad_replay_core,
    )



def crear_env_factory(
    *,
    rank: int,
    seed_base: int,
    configuracion_actual: ConfiguracionGeneradorInstancias,
    configuracion_core: ConfiguracionGeneradorInstancias,
    etapa: EtapaEntrenamientoRL,
    configuracion_reward: ConfiguracionRewardRL,
    configuracion_temporal: ConfiguracionPoliticaRL,
    catalogo_real: CatalogoDemandaReal | None,
    semillas_fijas: list[int] | None = None,
) -> Callable[[], gym.Env]:
    def _crear() -> gym.Env:
        generador = crear_generador_mixto(
            configuracion_actual=configuracion_actual,
            configuracion_core=configuracion_core,
            etapa=etapa,
            catalogo_real=catalogo_real,
        )
        env = EntornoEntrenamientoRL(
            generador=generador,
            seed_base=seed_base + rank * 100_000,
            semillas_fijas=semillas_fijas,
            max_pedidos=30,
            configuracion_reward=configuracion_reward,
            configuracion_temporal=configuracion_temporal,
        )
        return Monitor(env, info_keywords=INFO_KEYWORDS)

    return _crear



def crear_modelo(
    env: DummyVecEnv,
    seed: int,
    learning_rate: float,
) -> MaskablePPO:
    return MaskablePPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
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
        return [_normalizar_json(contenido) for contenido in valor]
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



def crear_validacion_fija(
    quick: bool,
) -> tuple[Any, Any, tuple[Any, ...], tuple[int, ...]]:
    b04 = next(
        caso.instancia
        for caso in crear_casos_benchmark_clasico()
        if caso.caso_id == "B04_VENTANAS"
    )
    proveedor = ProveedorVialCachePersistente(
        PYTHON_ROOT / "data" / "routing" / "cache_vial.csv",
        version_cache_esperada="pedemonte-vial-v1",
        permitir_fallback=False,
    )
    configuracion_base = ConfiguracionGeneradorInstancias(
        min_pedidos_finales=3,
        max_pedidos_finales=12,
        probabilidad_volcador=0.05,
        probabilidad_ventana_especifica=0.95,
        probabilidad_pedido_mayor_capacidad=0.0,
        ancho_ventana_min=45,
        ancho_ventana_max=120,
        modo_demanda_geografica=ModoDemandaGeografica.SINTETICA,
    )
    generador = GeneradorInstanciasPoliticaRL(
        GeneradorInstanciasRL(configuracion_base),
        ConfiguracionGeneradorPoliticaRL(
            probabilidad_patron_ventanas_conflictivas=0.90
        ),
    )
    cantidad = 24 if quick else 60
    semillas = tuple(range(265_000, 265_000 + cantidad))
    instancias = tuple(generador.generar(seed) for seed in semillas)
    return b04, proveedor, instancias, semillas



def main() -> None:
    args = parse_args()
    modelo_inicial = (
        None
        if args.from_scratch
        else Path(args.initial_model).expanduser().resolve()
    )
    if modelo_inicial is not None and not modelo_inicial.is_file():
        raise FileNotFoundError(
            f"No existe el checkpoint inicial: {modelo_inicial}"
        )

    etapas = (
        crear_curriculum_entrenamiento_rapido()
        if args.quick
        else crear_curriculum_entrenamiento()
    )
    run_dir = PYTHON_ROOT / "rl_artifacts" / args.run_name.strip()
    run_dir.mkdir(parents=True, exist_ok=True)

    configuracion_reward = ConfiguracionRewardRL(
        modo=ModoRewardRL.VENTAJA_GREEDY_RELATIVA,
        denominador_relativo_minimo=1.0,
    )
    configuracion_temporal = ConfiguracionPoliticaRL()
    catalogo_real = preparar_catalogo_real(args, etapas[0])
    b04, proveedor_b04, validacion_sintetica, semillas_validacion = (
        crear_validacion_fija(args.quick)
    )

    guardar_json(
        run_dir / "policy_config.json",
        {
            "version_entorno": "pedemonte-rl-policy",
            "modelo_inicial": modelo_inicial,
            "learning_rate": args.learning_rate,
            "run_name": args.run_name,
            "seed": args.seed,
            "n_envs": args.n_envs,
            "quick": args.quick,
            "demand_mode": args.demand_mode,
            "demand_dataset": args.demand_dataset,
            "reward_base": asdict(configuracion_reward),
            "temporal": asdict(configuracion_temporal),
            "curriculum": [asdict(etapa) for etapa in etapas],
            "validacion_externa": {
                "incluye_b04_cache_vial_estricta": True,
                "semillas_sinteticas": semillas_validacion,
                "seleccion_lexicografica": [
                    "b04_pedidos_tardios",
                    "b04_tardanza",
                    "maximizar_sinteticos_sin_riesgo",
                    "tardanza_sintetica_total",
                    "gap_costo_mediano_vs_greedy",
                ],
            },
            "continuacion_entre_etapas": "EXTERNAL_BEST",
            "modelo_productivo_modificado": False,
        },
    )

    print("=== ENTRENAMIENTO DE POLÍTICA RL ===")
    print(f"Directorio: {run_dir}")
    print(f"Seed: {args.seed}")
    print(f"Entornos: {args.n_envs}")
    print(f"Currículo rápido: {args.quick}")
    print("Selección entre etapas: EXTERNAL_BEST")
    print(f"Modelo inicial: {modelo_inicial or 'DESDE_CERO'}")
    print(f"Learning rate: {args.learning_rate}")
    print("Modelo productivo: NO SE MODIFICA")

    model: MaskablePPO | None = None
    selected_model_path: Path | None = None
    train_env: DummyVecEnv | None = None
    eval_env: DummyVecEnv | None = None

    try:
        for indice, etapa in enumerate(etapas, start=1):
            stage_dir = run_dir / etapa.nombre
            checkpoint_dir = stage_dir / "checkpoints"
            best_dir = stage_dir / "internal_best"
            evaluation_dir = stage_dir / "internal_evaluation"

            for directorio in (
                checkpoint_dir,
                best_dir,
                evaluation_dir,
            ):
                directorio.mkdir(parents=True, exist_ok=True)

            particion_train = (
                ParticionDemandaReal.ENTRENAMIENTO
                if args.demand_mode == ModoDemandaGeografica.REAL.value
                else None
            )
            particion_eval = (
                ParticionDemandaReal.VALIDACION
                if args.demand_mode == ModoDemandaGeografica.REAL.value
                else None
            )
            configuracion_train = crear_configuracion_generador(
                args,
                etapa,
                particion_train,
            )
            configuracion_eval = crear_configuracion_generador(
                args,
                etapa,
                particion_eval,
            )
            configuracion_core_train = crear_configuracion_core(
                args,
                particion_train,
            )
            configuracion_core_eval = crear_configuracion_core(
                args,
                particion_eval,
            )

            guardar_json(
                stage_dir / "stage_config.json",
                {
                    "indice": indice,
                    "etapa": asdict(etapa),
                    "generador_train": asdict(configuracion_train),
                    "generador_eval": asdict(configuracion_eval),
                    "generador_core_train": asdict(
                        configuracion_core_train
                    ),
                    "generador_core_eval": asdict(
                        configuracion_core_eval
                    ),
                    "modelo_inicio": selected_model_path,
                },
            )

            seed_etapa = args.seed + indice * 1_000_000
            train_env = DummyVecEnv(
                [
                    crear_env_factory(
                        rank=rank,
                        seed_base=seed_etapa,
                        configuracion_actual=configuracion_train,
                        configuracion_core=configuracion_core_train,
                        etapa=etapa,
                        configuracion_reward=configuracion_reward,
                        configuracion_temporal=configuracion_temporal,
                        catalogo_real=catalogo_real,
                    )
                    for rank in range(args.n_envs)
                ]
            )
            semillas_eval = list(
                range(
                    266_000 + indice * 1_000,
                    266_030 + indice * 1_000,
                )
            )
            eval_env = DummyVecEnv(
                [
                    crear_env_factory(
                        rank=0,
                        seed_base=semillas_eval[0],
                        configuracion_actual=configuracion_eval,
                        configuracion_core=configuracion_core_eval,
                        etapa=etapa,
                        configuracion_reward=configuracion_reward,
                        configuracion_temporal=configuracion_temporal,
                        catalogo_real=catalogo_real,
                        semillas_fijas=semillas_eval,
                    )
                ]
            )

            if model is None:
                if modelo_inicial is None:
                    model = crear_modelo(
                        train_env,
                        args.seed,
                        args.learning_rate,
                    )
                else:
                    model = MaskablePPO.load(
                        str(modelo_inicial),
                        env=train_env,
                        custom_objects={
                            "learning_rate": args.learning_rate,
                        },
                    )
            elif selected_model_path is not None:
                model = MaskablePPO.load(
                    str(selected_model_path),
                    env=train_env,
                    custom_objects={
                        "learning_rate": args.learning_rate,
                    },
                )
            else:
                model.set_env(train_env)

            callback_externo = ValidacionExternaCallback(
                eval_freq=max(etapa.eval_freq // args.n_envs, 1),
                stage_dir=stage_dir,
                instancia_b04=b04,
                proveedor_b04=proveedor_b04,
                instancias_sinteticas=validacion_sintetica,
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
                        name_prefix="rl_policy",
                        verbose=1,
                    ),
                    MaskableEvalCallback(
                        eval_env,
                        best_model_save_path=str(best_dir),
                        log_path=str(evaluation_dir),
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
                "Pedidos: "
                f"{etapa.min_pedidos_finales}-"
                f"{etapa.max_pedidos_finales}"
            )
            print(f"Timesteps: {etapa.timesteps}")
            print(
                "Patrón conflictivo: "
                f"{etapa.probabilidad_patron_conflictivo:.0%}"
            )
            print(
                "Replay core: "
                f"{etapa.probabilidad_replay_core:.0%}"
            )

            # Registra el modelo de entrada. Si la etapa empeora desde el
            # primer bloque PPO, este baseline sigue disponible para
            # revertir y continuar la etapa siguiente.
            callback_externo.evaluar_modelo_y_guardar(
                model,
                int(model.num_timesteps),
            )

            model.learn(
                total_timesteps=etapa.timesteps,
                callback=callbacks,
                reset_num_timesteps=False,
            )
            model.save(str(stage_dir / "final_model"))

            # Garantiza una evaluación externa del estado final aun si
            # el redondeo PPO no coincide exactamente con eval_freq.
            callback_externo.evaluar_modelo_y_guardar(
                model,
                int(model.num_timesteps),
            )
            selected_model_path = callback_externo.best_model_path

            if not selected_model_path.exists():
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

        if selected_model_path is None or not selected_model_path.exists():
            raise RuntimeError("No se seleccionó un modelo de la política RL.")

        ruta_final = run_dir / "final_model.zip"
        shutil.copy2(selected_model_path, ruta_final)
        guardar_json(
            run_dir / "final_model_selection.json",
            {
                "modelo_final": ruta_final,
                "origen_external_best": selected_model_path,
                "criterio": "VALIDACION_EXTERNA_LEXICOGRAFICA",
                "modelo_promovido": False,
            },
        )

        print("")
        print("Entrenamiento de la política RL finalizado.")
        print(f"Modelo final seleccionado: {ruta_final}")

        print("El modelo productivo no fue modificado.")

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
