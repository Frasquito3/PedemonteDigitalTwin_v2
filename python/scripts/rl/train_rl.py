"""Script unificado y reproducible de entrenamiento de políticas RL mediante archivos JSON.

Permite entrenar un modelo RL (desde cero o por transferencia de pesos) leyendo todos
los hiperparámetros, etapas de currículum y semillas desde un archivo de configuración JSON.
Guarda los artefactos versionados en `python/models/rl/historial/` y, si está configurado,
actualiza el modelo operativo activo en `python/models/rl/rl_policy.zip` y `rl_policies.json`.

Uso por línea de comandos:
    python python/scripts/rl/train_rl.py --config python/scripts/rl/configs/rl_v2_35slots.json
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from sb3_contrib import MaskablePPO

PYTHON_ROOT = Path(__file__).resolve().parents[2]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from planner.algorithms.greedy import generar_plan_greedy
from planner.core.config import ConfiguracionPlanificacion
from planner.core.schema import InstanciaTurno, PedidoInput, Turno
from planner.domain.preprocess import preprocesar_pedidos
from planner.rl.policy_config import ConfiguracionPoliticaRL
from planner.rl.policy_env import EntornoPlanificacionRL
from planner.rl.policy_planner import (
    PlanificadorPoliticaRL,
    _asegurar_compatibilidad_numpy,
)
from planner.rl.policy_runtime import VERSION_RUNTIME, PlanificadorOperativoRL
from planner.rl.reward import ConfiguracionRewardRL, ModoRewardRL
from planner.routing.objective import evaluar_plan_estimado
from planner.routing.travel import ProveedorHaversineAjustado, construir_matriz_viaje

from scripts.data_processing.generar_pedidos_dia import (
    DEFAULT_GEOCODING_CSV,
    DEFAULT_PROBABILIDADES_CSV,
    UbicacionHistorica,
    cargar_y_analizar_historico,
    generar_pedidos_para_un_dia,
)

MAX_PEDIDOS_V2 = 35
MODELS_DIR = PYTHON_ROOT / "models" / "rl"
HISTORIAL_DIR = MODELS_DIR / "historial"
CONFIGS_DIR = Path(__file__).resolve().parent / "configs"
DEFAULT_CONFIG_PATH = CONFIGS_DIR / "rl_v2_35slots.json"

__all__ = [
    "CONFIGS_DIR",
    "DEFAULT_CONFIG_PATH",
    "HISTORIAL_DIR",
    "MAX_PEDIDOS_V2",
    "MODELS_DIR",
    "EntornoCurricularMultiInstancia",
    "cargar_configuracion_entrenamiento",
    "construir_instancia_entrenamiento",
    "ejecutar_entrenamiento_desde_config",
    "ejecutar_entrenamiento_v2",
    "evaluar_lote_comparativo",
    "fijar_semillas_globales",
    "main",
    "respaldar_modelo_v1",
    "transferir_pesos_v1_a_v2",
]


def _sha256_archivo(ruta: Path) -> str:
    """Calcula el hash SHA-256 de un archivo binario en disco."""
    h = sha256()
    with ruta.open("rb") as f:
        for bloque in iter(lambda: f.read(1024 * 1024), b""):
            h.update(bloque)
    return h.hexdigest()


def fijar_semillas_globales(seed: int) -> None:
    """Fija las semillas de Python, NumPy y PyTorch para reproducibilidad determinista."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cargar_configuracion_entrenamiento(ruta_config: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Lee y valida el archivo JSON con los parámetros de entrenamiento del experimento."""
    ruta = Path(ruta_config).expanduser().resolve()
    if not ruta.is_file():
        raise FileNotFoundError(f"No existe el archivo de configuración RL: {ruta}")

    datos: dict[str, Any] = json.loads(ruta.read_text(encoding="utf-8"))
    for campo in (
        "experimento_id",
        "seed_global",
        "arquitectura",
        "inicializacion",
        "hiperparametros_ppo",
        "curriculum",
    ):
        if campo not in datos:
            raise ValueError(f"Falta el campo obligatorio '{campo}' en {ruta}")
    return datos


def respaldar_modelo_v1() -> tuple[Path, Path]:
    """Verifica o conserva una copia del modelo V1 en `models/rl/historial/`."""
    HISTORIAL_DIR.mkdir(parents=True, exist_ok=True)
    ruta_zip_v1 = HISTORIAL_DIR / "rl_policy_v1.zip"
    ruta_json_v1 = HISTORIAL_DIR / "rl_policies_v1.json"

    ruta_zip_actual = MODELS_DIR / "rl_policy.zip"
    ruta_json_actual = MODELS_DIR / "rl_policies.json"

    if not ruta_zip_v1.exists() and ruta_zip_actual.exists():
        shutil.copy2(ruta_zip_actual, ruta_zip_v1)

    if not ruta_json_v1.exists() and ruta_json_actual.exists():
        datos = json.loads(ruta_json_actual.read_text(encoding="utf-8"))
        datos["modelo"] = "rl_policy_v1.zip"
        datos["sha256_modelo"] = _sha256_archivo(ruta_zip_v1)
        ruta_json_v1.write_text(
            json.dumps(datos, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    return ruta_zip_v1, ruta_json_v1


def construir_instancia_entrenamiento(
    ubicaciones: list[UbicacionHistorica],
    rango_tareas: tuple[int, int],
    seed_escenario: int,
    max_pedidos: int = MAX_PEDIDOS_V2,
    *,
    prob_tarde: float = 0.45,
    prob_ventana: float = 0.55,
    escalar_camiones_por_tareas: bool = False,
) -> InstanciaTurno:
    """Genera una InstanciaTurno sintética y realista para etapas de entrenamiento RL."""
    rng = random.Random(seed_escenario)
    prob_manana = max(0.0, min(1.0, 1.0 - float(prob_tarde)))
    turno_str = "MANANA" if rng.random() < prob_manana else "TARDE"
    turno_enum = Turno.MANANA if turno_str == "MANANA" else Turno.TARDE
    inicio_min, fin_obj_min, fin_tol_min = (
        (450, 720, 735) if turno_enum == Turno.MANANA else (840, 1020, 1035)
    )

    tareas_objetivo = rng.randint(rango_tareas[0], rango_tareas[1])
    camiones_prom = (
        max(2.0, min(9.5, tareas_objetivo * 0.38)) if escalar_camiones_por_tareas else 6.0
    )
    pedidos_gen = generar_pedidos_para_un_dia(
        ubicaciones=ubicaciones,
        cantidad_pedidos=tareas_objetivo,
        turno=turno_str,
        dia_numero=1,
        prob_ventana=prob_ventana,
        camiones_turno_promedio=camiones_prom,
        rng=rng,
        max_tareas=tareas_objetivo,
    )

    originales: list[PedidoInput] = []
    mapa_volcador: dict[str, int] = {}
    for p in pedidos_gen:
        h_d, m_d = (int(x) for x in p.hora_desde.split(":"))
        h_h, m_h = (int(x) for x in p.hora_hasta.split(":"))
        req_v = p.requiere_volcador == "SI"
        mapa_volcador[p.pedido_id] = p.unidades if req_v else 0
        originales.append(
            PedidoInput(
                pedido_id=p.pedido_id,
                pedido_original_id=p.pedido_id,
                numero_parte=1,
                total_partes=1,
                turno=turno_enum,
                latitud=p.latitud,
                longitud=p.longitud,
                unidades_capacidad=p.unidades,
                requiere_volcador=req_v,
                tiene_ventana_especifica=(p.tiene_ventana == "SI"),
                hora_desde_min=h_d * 60 + m_d,
                hora_hasta_min=h_h * 60 + m_h,
            )
        )

    tareas = preprocesar_pedidos(
        originales,
        capacidad_camion=8,
        unidades_volcador_por_pedido=mapa_volcador,
    )
    tareas = tareas[:max_pedidos]

    return InstanciaTurno(
        instancia_id=f"TRAIN-{seed_escenario}",
        fecha_operacion="2026-10-30",
        turno=turno_enum,
        pedidos=tareas,
        lat_corralon=-32.8495006,
        lon_corralon=-60.722653,
        capacidad_camion=8,
        cantidad_camiones=2,
        hora_inicio_turno_min=inicio_min,
        hora_fin_objetivo_min=fin_obj_min,
        hora_fin_tolerancia_min=fin_tol_min,
        seed_escenario=seed_escenario,
        seed_ejecucion=seed_escenario + 100_000,
    )


class EntornoCurricularMultiInstancia(gym.Env[np.ndarray, int]):
    """Envuelve `EntornoPlanificacionRL` sorteando una instancia nueva en cada `reset()`."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        ubicaciones: list[UbicacionHistorica],
        rango_tareas: tuple[int, int],
        seed_base: int = 10_000,
        max_pedidos: int = MAX_PEDIDOS_V2,
        usar_mascara_temporal_dura: bool = True,
        ordenar_slots_canonicamente: bool = True,
        *,
        prob_tarde: float = 0.45,
        prob_ventana: float = 0.55,
        escalar_camiones_por_tareas: bool = False,
        modo_reward: ModoRewardRL = ModoRewardRL.ABSOLUTO,
        peso_terminal_costo_acotado: float = 0.50,
    ) -> None:
        super().__init__()
        self.ubicaciones = ubicaciones
        self.rango_tareas = rango_tareas
        self.rng = random.Random(seed_base)
        self.max_pedidos = max_pedidos
        self.prob_tarde = prob_tarde
        self.prob_ventana = prob_ventana
        self.escalar_camiones_por_tareas = escalar_camiones_por_tareas
        self.config_plan = ConfiguracionPlanificacion()
        self.proveedor = ProveedorHaversineAjustado()
        self.config_reward = ConfiguracionRewardRL(modo=modo_reward)
        self.config_pol = ConfiguracionPoliticaRL(
            usar_mascara_temporal_dura=usar_mascara_temporal_dura,
            ordenar_slots_canonicamente=ordenar_slots_canonicamente,
            peso_terminal_costo_acotado=peso_terminal_costo_acotado,
        )
        self._env_interno = self._crear_env_nuevo()
        self.action_space = self._env_interno.action_space
        self.observation_space = self._env_interno.observation_space

    def _crear_env_nuevo(self) -> EntornoPlanificacionRL:
        seed_esc = self.rng.randint(100_000, 999_999_999)
        instancia = construir_instancia_entrenamiento(
            self.ubicaciones,
            self.rango_tareas,
            seed_esc,
            max_pedidos=self.max_pedidos,
            prob_tarde=self.prob_tarde,
            prob_ventana=self.prob_ventana,
            escalar_camiones_por_tareas=self.escalar_camiones_por_tareas,
        )
        return EntornoPlanificacionRL(
            instancia=instancia,
            configuracion=self.config_plan,
            proveedor_viaje=self.proveedor,
            max_pedidos=self.max_pedidos,
            configuracion_reward=self.config_reward,
            configuracion_temporal=self.config_pol,
        )

    def action_masks(self) -> np.ndarray:
        """Devuelve la máscara de acciones factibles del entorno interno."""
        return self._env_interno.action_masks()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reinicia el entorno generando una nueva instancia estocástica."""
        if seed is not None:
            self.rng.seed(seed)
        self._env_interno = self._crear_env_nuevo()
        return self._env_interno.reset(seed=seed, options=options)

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Aplica la acción seleccionada en el entorno interno."""
        return self._env_interno.step(action)


def transferir_pesos_v1_a_v2(
    ruta_v1: Path,
    modelo_v2: MaskablePPO,
    max_pedidos_destino: int = MAX_PEDIDOS_V2,
    max_pedidos_origen: int = 30,
    slots_entrenados_origen: int = 12,
) -> None:
    """Transfiere los pesos de un checkpoint previo hacia la red extendida mediante simetría."""
    _asegurar_compatibilidad_numpy()
    planner_v1 = PlanificadorPoliticaRL(model_path=ruta_v1, max_pedidos=max_pedidos_origen)
    sd_v1 = planner_v1.model.policy.state_dict()
    sd_v2 = modelo_v2.policy.state_dict()

    f_por_ped = EntornoPlanificacionRL.FEATURES_POR_PEDIDO
    f_glob = EntornoPlanificacionRL.FEATURES_GLOBALES
    slots_base = min(slots_entrenados_origen, max_pedidos_origen, max_pedidos_destino)

    with torch.no_grad():
        for clave in (
            "mlp_extractor.policy_net.0.bias",
            "mlp_extractor.policy_net.2.weight",
            "mlp_extractor.policy_net.2.bias",
            "mlp_extractor.value_net.0.bias",
            "mlp_extractor.value_net.2.weight",
            "mlp_extractor.value_net.2.bias",
            "value_net.weight",
            "value_net.bias",
        ):
            if clave in sd_v1 and clave in sd_v2 and sd_v1[clave].shape == sd_v2[clave].shape:
                sd_v2[clave].copy_(sd_v1[clave])

        for clave_w in (
            "mlp_extractor.policy_net.0.weight",
            "mlp_extractor.value_net.0.weight",
        ):
            w_old = sd_v1[clave_w]
            w_new = sd_v2[clave_w]

            w_new[:, : slots_base * f_por_ped].copy_(w_old[:, : slots_base * f_por_ped])

            inicio_plantilla = min(4, max(0, slots_base - 1))
            plantilla_bloques = [
                w_old[:, s * f_por_ped : (s + 1) * f_por_ped]
                for s in range(inicio_plantilla, slots_base)
            ]
            bloque_promedio = torch.stack(plantilla_bloques, dim=0).mean(dim=0)

            for s_nuevo in range(slots_base, max_pedidos_destino):
                ref_bloque = plantilla_bloques[s_nuevo % len(plantilla_bloques)]
                mezcla = 0.70 * bloque_promedio + 0.30 * ref_bloque
                w_new[:, s_nuevo * f_por_ped : (s_nuevo + 1) * f_por_ped].copy_(mezcla * 0.65)

            w_new[
                :, max_pedidos_destino * f_por_ped : max_pedidos_destino * f_por_ped + f_glob
            ].copy_(
                w_old[:, max_pedidos_origen * f_por_ped : max_pedidos_origen * f_por_ped + f_glob]
            )

        act_w_old = sd_v1["action_net.weight"]
        act_b_old = sd_v1["action_net.bias"]
        act_w_new = sd_v2["action_net.weight"]
        act_b_new = sd_v2["action_net.bias"]

        act_w_new[:slots_base, :].copy_(act_w_old[:slots_base, :])
        act_b_new[:slots_base].copy_(act_b_old[:slots_base])

        inicio_plantilla = min(4, max(0, slots_base - 1))
        filas_plantilla = [act_w_old[s, :] for s in range(inicio_plantilla, slots_base)]
        fila_media = torch.stack(filas_plantilla, dim=0).mean(dim=0)
        bias_medio = act_b_old[inicio_plantilla:slots_base].mean()

        for s_nuevo in range(slots_base, max_pedidos_destino):
            ref_fila = filas_plantilla[s_nuevo % len(filas_plantilla)]
            act_w_new[s_nuevo, :].copy_(0.70 * fila_media + 0.30 * ref_fila)
            act_b_new[s_nuevo].copy_(bias_medio)

    modelo_v2.policy.load_state_dict(sd_v2)


def _evaluar_modelo_en_memoria(
    modelo: MaskablePPO,
    instancias_val: list[InstanciaTurno],
    config_pol: ConfiguracionPoliticaRL,
    max_pedidos: int,
    num_rollouts: int = 8,
) -> tuple[float, float, int]:
    """Evalúa el modelo actual en memoria sobre un lote de validación sin tocar el RNG global."""
    estado_py = random.getstate()
    estado_np = np.random.get_state()
    estado_torch = torch.get_rng_state()

    config_plan = ConfiguracionPlanificacion()
    proveedor = ProveedorHaversineAjustado()
    total_tardios = 0
    total_tardanza = 0.0
    total_costo = 0.0

    try:
        for inst in instancias_val:
            env = EntornoPlanificacionRL(
                instancia=inst,
                configuracion=config_plan,
                proveedor_viaje=proveedor,
                configuracion_temporal=config_pol,
                max_pedidos=max_pedidos,
            )
            try:
                seed_rl = inst.seed_ejecucion if inst.seed_ejecucion > 0 else inst.seed_escenario
                mejor_clave: tuple[float, float, int] | None = None
                for k in range(max(1, num_rollouts)):
                    determinista = k == 0
                    seed_ep = seed_rl + k * 997
                    if not determinista and hasattr(modelo, "set_random_seed"):
                        modelo.set_random_seed(seed_ep)
                    obs, _ = env.reset(seed=seed_ep)
                    terminado = False
                    while not terminado:
                        mascara = env.action_masks()
                        accion, _ = modelo.predict(
                            obs,
                            action_masks=mascara,
                            deterministic=determinista,
                        )
                        accion_int = int(np.asarray(accion).item())
                        obs, _, terminado, _, _ = env.step(accion_int)
                    plan_k = env.ultimo_plan
                    est_k = env.ultima_estimacion
                    costo_k = float(plan_k.costo_estimado) if plan_k is not None else 1e9
                    tardios_k = int(est_k.pedidos_tardios) if est_k is not None else 999
                    tardanza_k = float(est_k.tardanza_total_min) if est_k is not None else 1e9
                    clave_k = (costo_k, tardanza_k, tardios_k)
                    if mejor_clave is None or clave_k < mejor_clave:
                        mejor_clave = clave_k
                assert mejor_clave is not None
                total_costo += mejor_clave[0] / max(1, len(instancias_val))
                total_tardanza += mejor_clave[1]
                total_tardios += mejor_clave[2]
            finally:
                env.close()
    finally:
        random.setstate(estado_py)
        np.random.set_state(estado_np)
        torch.set_rng_state(estado_torch)

    return (round(total_costo, 4), round(total_tardanza, 4), total_tardios)


def evaluar_lote_comparativo(
    ubicaciones: list[UbicacionHistorica],
    ruta_manifiesto_v1: Path,
    ruta_manifiesto_v2: Path,
    config_eval: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compara el modelo de referencia (V1), el nuevo modelo entrenado, V2 previo (opcional) y GREEDY."""
    cfg = config_eval or {}
    n_turnos = int(cfg.get("cantidad_turnos_por_segmento", 20))
    seed_chicos = int(cfg.get("seed_inicio_turnos_chicos", 500_001))
    paso_chicos = int(cfg.get("paso_seed_turnos_chicos", 17))
    rango_chicos_raw = cfg.get("rango_tareas_chicos", [6, 12])
    rango_chicos = (int(rango_chicos_raw[0]), int(rango_chicos_raw[1]))

    seed_grandes = int(cfg.get("seed_inicio_turnos_grandes", 700_001))
    paso_grandes = int(cfg.get("paso_seed_turnos_grandes", 29))
    rango_grandes_raw = cfg.get("rango_tareas_grandes", [14, 32])
    rango_grandes = (int(rango_grandes_raw[0]), int(rango_grandes_raw[1]))

    config = ConfiguracionPlanificacion()
    proveedor = ProveedorHaversineAjustado()

    planner_v1 = PlanificadorOperativoRL(
        manifest_path=ruta_manifiesto_v1,
        configuracion=config,
        proveedor_viaje=proveedor,
        max_pedidos=30,
    )
    planner_v2 = PlanificadorOperativoRL(
        manifest_path=ruta_manifiesto_v2,
        configuracion=config,
        proveedor_viaje=proveedor,
        max_pedidos=MAX_PEDIDOS_V2,
    )

    planner_prev: PlanificadorOperativoRL | None = None
    ref_prev_rel = cfg.get("manifiesto_v2_previo")
    if ref_prev_rel:
        ruta_prev = (PYTHON_ROOT / str(ref_prev_rel)).resolve()
        if ruta_prev.is_file() and ruta_prev != ruta_manifiesto_v2.resolve():
            planner_prev = PlanificadorOperativoRL(
                manifest_path=ruta_prev,
                configuracion=config,
                proveedor_viaje=proveedor,
                max_pedidos=MAX_PEDIDOS_V2,
            )

    semillas_chicas = [seed_chicos + i * paso_chicos for i in range(n_turnos)]
    res_chicos: dict[str, dict[str, Any]] = {
        "RL_V1": {"tardios": 0, "tardanza_min": 0.0, "costo_promedio": 0.0, "exitos": 0},
        "RL_V2": {"tardios": 0, "tardanza_min": 0.0, "costo_promedio": 0.0, "exitos": 0},
        "GREEDY": {"tardios": 0, "tardanza_min": 0.0, "costo_promedio": 0.0, "exitos": 0},
    }
    if planner_prev is not None:
        res_chicos["RL_V2_PREVIO"] = {
            "tardios": 0,
            "tardanza_min": 0.0,
            "costo_promedio": 0.0,
            "exitos": 0,
        }

    victorias_vs_prev = 0
    empates_vs_prev = 0
    derrotas_vs_prev = 0

    for s in semillas_chicas:
        inst = construir_instancia_entrenamiento(ubicaciones, rango_chicos, s)
        matriz = construir_matriz_viaje(inst, config, proveedor)

        metodos_chicos: list[tuple[str, Any]] = [
            ("RL_V1", planner_v1.generar_plan),
            ("RL_V2", planner_v2.generar_plan),
            (
                "GREEDY",
                lambda x: generar_plan_greedy(x, configuracion=config, proveedor_viaje=proveedor),
            ),
        ]
        if planner_prev is not None:
            metodos_chicos.append(("RL_V2_PREVIO", planner_prev.generar_plan))

        costos_turno: dict[str, float] = {}
        for nombre, fn in metodos_chicos:
            plan = fn(inst)
            est = evaluar_plan_estimado(inst, plan, matriz, config)
            res_chicos[nombre]["tardios"] += int(est.pedidos_tardios)
            res_chicos[nombre]["tardanza_min"] += float(est.tardanza_total_min)
            res_chicos[nombre]["costo_promedio"] += float(est.costo_total) / len(semillas_chicas)
            res_chicos[nombre]["exitos"] += 1
            costos_turno[nombre] = float(est.costo_total)

        if "RL_V2_PREVIO" in costos_turno:
            diff = costos_turno["RL_V2"] - costos_turno["RL_V2_PREVIO"]
            if diff < -1e-6:
                victorias_vs_prev += 1
            elif diff > 1e-6:
                derrotas_vs_prev += 1
            else:
                empates_vs_prev += 1

    semillas_grandes = [seed_grandes + i * paso_grandes for i in range(n_turnos)]
    res_grandes: dict[str, dict[str, Any]] = {
        "RL_V1": {"ejecutables": 0, "fallos_por_tope_12": 0},
        "RL_V2": {
            "ejecutables": 0,
            "tardios": 0,
            "tardanza_min": 0.0,
            "costo_promedio": 0.0,
        },
        "GREEDY": {
            "ejecutables": 0,
            "tardios": 0,
            "tardanza_min": 0.0,
            "costo_promedio": 0.0,
        },
    }
    if planner_prev is not None:
        res_grandes["RL_V2_PREVIO"] = {
            "ejecutables": 0,
            "tardios": 0,
            "tardanza_min": 0.0,
            "costo_promedio": 0.0,
        }

    for s in semillas_grandes:
        inst = construir_instancia_entrenamiento(ubicaciones, rango_grandes, s)
        matriz = construir_matriz_viaje(inst, config, proveedor)

        try:
            planner_v1.generar_plan(inst)
            res_grandes["RL_V1"]["ejecutables"] += 1
        except RuntimeError:
            res_grandes["RL_V1"]["fallos_por_tope_12"] += 1

        plan_v2 = planner_v2.generar_plan(inst)
        est_v2 = evaluar_plan_estimado(inst, plan_v2, matriz, config)
        res_grandes["RL_V2"]["ejecutables"] += 1
        res_grandes["RL_V2"]["tardios"] += int(est_v2.pedidos_tardios)
        res_grandes["RL_V2"]["tardanza_min"] += float(est_v2.tardanza_total_min)
        res_grandes["RL_V2"]["costo_promedio"] += float(est_v2.costo_total) / len(semillas_grandes)

        if planner_prev is not None:
            plan_prev = planner_prev.generar_plan(inst)
            est_prev = evaluar_plan_estimado(inst, plan_prev, matriz, config)
            res_grandes["RL_V2_PREVIO"]["ejecutables"] += 1
            res_grandes["RL_V2_PREVIO"]["tardios"] += int(est_prev.pedidos_tardios)
            res_grandes["RL_V2_PREVIO"]["tardanza_min"] += float(est_prev.tardanza_total_min)
            res_grandes["RL_V2_PREVIO"]["costo_promedio"] += float(est_prev.costo_total) / len(
                semillas_grandes
            )
            diff_g = float(est_v2.costo_total) - float(est_prev.costo_total)
            if diff_g < -1e-6:
                victorias_vs_prev += 1
            elif diff_g > 1e-6:
                derrotas_vs_prev += 1
            else:
                empates_vs_prev += 1

        plan_gr = generar_plan_greedy(inst, configuracion=config, proveedor_viaje=proveedor)
        est_gr = evaluar_plan_estimado(inst, plan_gr, matriz, config)
        res_grandes["GREEDY"]["ejecutables"] += 1
        res_grandes["GREEDY"]["tardios"] += int(est_gr.pedidos_tardios)
        res_grandes["GREEDY"]["tardanza_min"] += float(est_gr.tardanza_total_min)
        res_grandes["GREEDY"]["costo_promedio"] += float(est_gr.costo_total) / len(semillas_grandes)

    salida: dict[str, Any] = {
        "turnos_6_a_12_tareas": res_chicos,
        "turnos_14_a_32_tareas": res_grandes,
    }
    if planner_prev is not None:
        total_eval = len(semillas_chicas) + len(semillas_grandes)
        salida["comparacion_pareada_vs_v2_previo"] = {
            "turnos_totales": total_eval,
            "victorias_nuevo": victorias_vs_prev,
            "empates": empates_vs_prev,
            "derrotas_nuevo": derrotas_vs_prev,
            "tasa_no_inferioridad_pct": round(
                100.0 * (victorias_vs_prev + empates_vs_prev) / max(1, total_eval), 2
            ),
        }
    return salida


def ejecutar_entrenamiento_desde_config(
    ruta_config: Path = DEFAULT_CONFIG_PATH,
    activar_override: bool | None = None,
) -> dict[str, Any]:
    """Ejecuta el entrenamiento completo de un modelo RL a partir de un archivo JSON."""
    config = cargar_configuracion_entrenamiento(ruta_config)
    exp_id = str(config["experimento_id"]).strip()
    seed_global = int(config["seed_global"])
    fijar_semillas_globales(seed_global)

    arq = config["arquitectura"]
    max_pedidos = int(arq.get("max_pedidos", MAX_PEDIDOS_V2))
    usar_mascara = bool(arq.get("usar_mascara_temporal_dura", True))
    orden_canonico = bool(arq.get("ordenar_slots_canonicamente", True))
    net_pi = [int(x) for x in arq.get("net_arch_pi", [256, 256])]
    net_vf = [int(x) for x in arq.get("net_arch_vf", [256, 256])]
    modo_reward_str = str(arq.get("modo_reward", "ABSOLUTO")).strip().upper()
    modo_reward_enum = (
        ModoRewardRL.VENTAJA_GREEDY_RELATIVA
        if modo_reward_str == "VENTAJA_GREEDY_RELATIVA"
        else ModoRewardRL.ABSOLUTO
    )
    peso_costo_terminal = float(arq.get("peso_terminal_costo_acotado", 0.50))

    ppo_cfg = config["hiperparametros_ppo"]
    init_cfg = config["inicializacion"]
    modo_init = str(init_cfg.get("modo", "desde_cero")).strip().lower()
    curriculum = list(config["curriculum"])
    if not curriculum:
        raise ValueError("El bloque 'curriculum' debe tener al menos una etapa.")

    HISTORIAL_DIR.mkdir(parents=True, exist_ok=True)
    ubicaciones = cargar_y_analizar_historico(
        DEFAULT_GEOCODING_CSV,
        ruta_probabilidades=DEFAULT_PROBABILIDADES_CSV,
    )

    primera_etapa = curriculum[0]
    rango_primero = (
        int(primera_etapa["rango_tareas"][0]),
        int(primera_etapa["rango_tareas"][1]),
    )
    env_inicial = EntornoCurricularMultiInstancia(
        ubicaciones=ubicaciones,
        rango_tareas=rango_primero,
        seed_base=int(primera_etapa["seed_entorno"]),
        max_pedidos=max_pedidos,
        usar_mascara_temporal_dura=usar_mascara,
        ordenar_slots_canonicamente=orden_canonico,
        prob_tarde=float(primera_etapa.get("prob_tarde", 0.45)),
        prob_ventana=float(primera_etapa.get("prob_ventana", 0.55)),
        escalar_camiones_por_tareas=bool(primera_etapa.get("escalar_camiones_por_tareas", False)),
        modo_reward=modo_reward_enum,
        peso_terminal_costo_acotado=peso_costo_terminal,
    )

    lr_ini = float(ppo_cfg.get("learning_rate", 1.5e-5))
    lr_tipo = str(ppo_cfg.get("lr_schedule", "constante")).strip().lower()
    if lr_tipo == "lineal":
        lr_fin = float(ppo_cfg.get("learning_rate_final", lr_ini * 0.25))

        def lr_schedule_fn(prog: float | int) -> float:
            return lr_fin + float(prog) * (lr_ini - lr_fin)

        lr_param: Any = lr_schedule_fn
    else:
        lr_param = lr_ini

    modelo = MaskablePPO(
        "MlpPolicy",
        env_inicial,
        learning_rate=lr_param,
        n_steps=int(ppo_cfg.get("n_steps", 256)),
        batch_size=int(ppo_cfg.get("batch_size", 128)),
        n_epochs=int(ppo_cfg.get("n_epochs", 6)),
        gamma=float(ppo_cfg.get("gamma", 1.0)),
        gae_lambda=float(ppo_cfg.get("gae_lambda", 0.95)),
        clip_range=float(ppo_cfg.get("clip_range", 0.2)),
        ent_coef=float(ppo_cfg.get("ent_coef", 0.008)),
        policy_kwargs={"net_arch": dict(pi=net_pi, vf=net_vf)},
        seed=seed_global,
        verbose=0,
    )

    sha256_base: str | None = None
    ruta_base_rel: str | None = None
    if modo_init == "transfer_learning":
        modelo_base_str = str(init_cfg.get("modelo_base", "models/rl/historial/rl_policy_v1.zip"))
        ruta_base = (PYTHON_ROOT / modelo_base_str).resolve()
        if not ruta_base.is_file():
            ruta_base, _ = respaldar_modelo_v1()
        sha256_base = _sha256_archivo(ruta_base)
        ruta_base_rel = modelo_base_str
        transferir_pesos_v1_a_v2(
            ruta_v1=ruta_base,
            modelo_v2=modelo,
            max_pedidos_destino=max_pedidos,
            max_pedidos_origen=int(init_cfg.get("max_pedidos_modelo_base", 30)),
            slots_entrenados_origen=int(init_cfg.get("slots_entrenados_modelo_base", 12)),
        )
        print(f"[OK] Pesos transferidos desde {ruta_base.name} a {max_pedidos} slots.")

    eval_cfg = config.get("evaluacion", {})
    seleccionar_mejor = bool(eval_cfg.get("seleccionar_mejor_etapa", False))
    inf_cfg = config.get("inferencia_operativa", {})
    num_rollouts = int(inf_cfg.get("num_rollouts", 16))
    config_pol_val = ConfiguracionPoliticaRL(
        usar_mascara_temporal_dura=usar_mascara,
        ordenar_slots_canonicamente=orden_canonico,
    )

    instancias_val_etapa: list[InstanciaTurno] = []
    mejor_puntaje_etapa: tuple[float, float, int] | None = None
    mejor_state_dict: dict[str, torch.Tensor] | None = None
    mejor_nombre_etapa = "final"
    historial_etapas: list[dict[str, Any]] = []

    if seleccionar_mejor:
        estado_py = random.getstate()
        try:
            n_val_chicos = int(eval_cfg.get("cantidad_turnos_por_segmento", 20))
            s_ch = int(eval_cfg.get("seed_inicio_turnos_chicos", 500_001))
            p_ch = int(eval_cfg.get("paso_seed_turnos_chicos", 17))
            r_ch = (
                int(eval_cfg.get("rango_tareas_chicos", [6, 12])[0]),
                int(eval_cfg.get("rango_tareas_chicos", [6, 12])[1]),
            )
            s_gr = int(eval_cfg.get("seed_inicio_turnos_grandes", 700_001))
            p_gr = int(eval_cfg.get("paso_seed_turnos_grandes", 29))
            r_gr = (
                int(eval_cfg.get("rango_tareas_grandes", [14, 32])[0]),
                int(eval_cfg.get("rango_tareas_grandes", [14, 32])[1]),
            )
            for i in range(n_val_chicos):
                instancias_val_etapa.append(
                    construir_instancia_entrenamiento(ubicaciones, r_ch, s_ch + i * p_ch)
                )
            for i in range(n_val_chicos):
                instancias_val_etapa.append(
                    construir_instancia_entrenamiento(ubicaciones, r_gr, s_gr + i * p_gr)
                )
        finally:
            random.setstate(estado_py)

        puntaje_ini = _evaluar_modelo_en_memoria(
            modelo=modelo,
            instancias_val=instancias_val_etapa,
            config_pol=config_pol_val,
            max_pedidos=max_pedidos,
            num_rollouts=num_rollouts,
        )
        mejor_puntaje_etapa = puntaje_ini
        mejor_state_dict = {
            k: v.detach().cpu().clone() for k, v in modelo.policy.state_dict().items()
        }
        mejor_nombre_etapa = "etapa_0_base_transferida"
        historial_etapas.append(
            {
                "etapa": mejor_nombre_etapa,
                "costo_promedio": puntaje_ini[0],
                "tardanza_min": puntaje_ini[1],
                "tardios": puntaje_ini[2],
            }
        )
        print(
            f"   [VAL] {mejor_nombre_etapa}: costo_prom={puntaje_ini[0]:.2f}, "
            f"tardanza={puntaje_ini[1]:.2f}m, tardios={puntaje_ini[2]}"
        )

    for idx, etapa in enumerate(curriculum):
        nombre_etapa = str(etapa.get("nombre", f"etapa_{idx + 1}"))
        rango = (int(etapa["rango_tareas"][0]), int(etapa["rango_tareas"][1]))
        timesteps = int(etapa["timesteps"])
        seed_env = int(etapa["seed_entorno"])

        if idx > 0:
            env_etapa = EntornoCurricularMultiInstancia(
                ubicaciones=ubicaciones,
                rango_tareas=rango,
                seed_base=seed_env,
                max_pedidos=max_pedidos,
                usar_mascara_temporal_dura=usar_mascara,
                ordenar_slots_canonicamente=orden_canonico,
                prob_tarde=float(etapa.get("prob_tarde", 0.45)),
                prob_ventana=float(etapa.get("prob_ventana", 0.55)),
                escalar_camiones_por_tareas=bool(etapa.get("escalar_camiones_por_tareas", False)),
                modo_reward=modo_reward_enum,
                peso_terminal_costo_acotado=peso_costo_terminal,
            )
            modelo.set_env(env_etapa)

        print(f"-> Entrenando {nombre_etapa} ({rango[0]}..{rango[1]} tareas, {timesteps} pasos)...")
        modelo.learn(total_timesteps=timesteps, reset_num_timesteps=(idx == 0))

        if seleccionar_mejor and instancias_val_etapa:
            puntaje_et = _evaluar_modelo_en_memoria(
                modelo=modelo,
                instancias_val=instancias_val_etapa,
                config_pol=config_pol_val,
                max_pedidos=max_pedidos,
                num_rollouts=num_rollouts,
            )
            historial_etapas.append(
                {
                    "etapa": nombre_etapa,
                    "costo_promedio": puntaje_et[0],
                    "tardanza_min": puntaje_et[1],
                    "tardios": puntaje_et[2],
                }
            )
            print(
                f"   [VAL] {nombre_etapa}: costo_prom={puntaje_et[0]:.2f}, "
                f"tardanza={puntaje_et[1]:.2f}m, tardios={puntaje_et[2]}"
            )
            if mejor_puntaje_etapa is None or puntaje_et <= mejor_puntaje_etapa:
                mejor_puntaje_etapa = puntaje_et
                mejor_state_dict = {
                    k: v.detach().cpu().clone() for k, v in modelo.policy.state_dict().items()
                }
                mejor_nombre_etapa = nombre_etapa

    if seleccionar_mejor and mejor_state_dict is not None:
        modelo.policy.load_state_dict(mejor_state_dict)
        print(f"[OK] Mejor checkpoint seleccionado: {mejor_nombre_etapa} ({mejor_puntaje_etapa})")

    nombre_zip_hist = f"rl_policy_{exp_id}.zip"
    nombre_json_hist = f"rl_policies_{exp_id}.json"
    ruta_zip_hist = HISTORIAL_DIR / nombre_zip_hist
    ruta_json_hist = HISTORIAL_DIR / nombre_json_hist

    modelo.save(str(ruta_zip_hist))
    hash_modelo = _sha256_archivo(ruta_zip_hist)

    inf_cfg = config.get("inferencia_operativa", {})
    num_rollouts = int(inf_cfg.get("num_rollouts", 16))

    try:
        config_rel = str(Path(ruta_config).resolve().relative_to(PYTHON_ROOT)).replace("\\", "/")
    except ValueError:
        config_rel = str(ruta_config)

    bloque_entrenamiento: dict[str, Any] = {
        "archivo_config": config_rel,
        "experimento_id": exp_id,
        "seed_global": seed_global,
        "modo_inicializacion": modo_init,
        "modelo_base": ruta_base_rel,
        "sha256_modelo_base": sha256_base,
        "hiperparametros_ppo": ppo_cfg,
        "curriculum": curriculum,
    }
    if seleccionar_mejor and historial_etapas:
        bloque_entrenamiento["mejor_etapa_seleccionada"] = mejor_nombre_etapa
        bloque_entrenamiento["historial_validacion_etapas"] = historial_etapas

    manifiesto_hist: dict[str, Any] = {
        "version": VERSION_RUNTIME,
        "modelo": nombre_zip_hist,
        "modelo_respaldo_v1": "rl_policy_v1.zip",
        "sha256_modelo": hash_modelo,
        "max_pedidos_rl_validado": max_pedidos,
        "usar_mascara_temporal_dura": usar_mascara,
        "ordenar_slots_canonicamente": orden_canonico,
        "num_rollouts": num_rollouts,
        "configuracion_entrenamiento": bloque_entrenamiento,
    }

    ruta_json_hist.write_text(
        json.dumps(manifiesto_hist, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    eval_cfg = config.get("evaluacion", {})
    comparacion: dict[str, Any] = {}
    if bool(eval_cfg.get("ejecutar_comparacion", True)):
        ref_rel = str(
            eval_cfg.get("manifiesto_referencia", "models/rl/historial/rl_policies_v1.json")
        )
        ruta_manifiesto_ref = (PYTHON_ROOT / ref_rel).resolve()
        if ruta_manifiesto_ref.is_file():
            comparacion = evaluar_lote_comparativo(
                ubicaciones=ubicaciones,
                ruta_manifiesto_v1=ruta_manifiesto_ref,
                ruta_manifiesto_v2=ruta_json_hist,
                config_eval=eval_cfg,
            )
            manifiesto_hist["validacion_comparativa_v1_vs_v2"] = comparacion
            ruta_json_hist.write_text(
                json.dumps(manifiesto_hist, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

    activar = (
        bool(config.get("activar_como_modelo_actual", True))
        if activar_override is None
        else bool(activar_override)
    )
    if activar:
        ruta_zip_activo = MODELS_DIR / "rl_policy.zip"
        ruta_json_activo = MODELS_DIR / "rl_policies.json"
        shutil.copy2(ruta_zip_hist, ruta_zip_activo)
        manifiesto_activo: dict[str, Any] = dict(manifiesto_hist)
        manifiesto_activo["modelo"] = "rl_policy.zip"
        manifiesto_activo["modelo_respaldo_v1"] = "historial/rl_policy_v1.zip"
        ruta_json_activo.write_text(
            json.dumps(manifiesto_activo, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print("[OK] Modelo activado en models/rl/rl_policy.zip y models/rl/rl_policies.json")

    print(f"[OK] Artefactos guardados en historial: {ruta_zip_hist.name} y {ruta_json_hist.name}")
    return comparacion


def ejecutar_entrenamiento_v2() -> dict[str, Any]:
    """Ejecuta el entrenamiento del modelo V2 usando `configs/rl_v2_35slots.json`."""
    return ejecutar_entrenamiento_desde_config(DEFAULT_CONFIG_PATH)


def _crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Entrena un modelo RL a partir de un archivo de configuración JSON."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Ruta al archivo JSON con los parámetros del experimento.",
    )
    grupo_act = parser.add_mutually_exclusive_group()
    grupo_act.add_argument(
        "--activar",
        action="store_true",
        help="Fuerza activar el modelo entrenado como rl_policy.zip en models/rl/.",
    )
    grupo_act.add_argument(
        "--no-activar",
        action="store_true",
        help="Guarda el modelo solo en models/rl/historial/ sin pisar rl_policy.zip.",
    )
    return parser


def main() -> int:
    """Punto de entrada CLI para entrenamiento reproducible de políticas RL."""
    args = _crear_parser().parse_args()
    activar_override: bool | None = None
    if args.activar:
        activar_override = True
    elif args.no_activar:
        activar_override = False

    ejecutar_entrenamiento_desde_config(
        ruta_config=args.config,
        activar_override=activar_override,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
