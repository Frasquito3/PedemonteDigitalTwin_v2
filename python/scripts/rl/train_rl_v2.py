"""
Entrenamiento y evaluación del modelo RL mejorado (RL_V2, hasta 35 tareas)
preservando el modelo original como `rl_policy_v1.zip` / `rl_policies_v1.json`
y dejando configurado `rl_policy.zip` / `rl_policies.json` para AnyLogic.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import random
import shutil
import sys
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
from planner.rl.policy_runtime import PlanificadorOperativoRL, VERSION_RUNTIME
from planner.routing.objective import evaluar_plan_estimado
from planner.routing.travel import ProveedorHaversineAjustado, construir_matriz_viaje
from scripts.data_processing.generar_pedidos_dia import (
    DEFAULT_CANT_PEDIDOS_CSV,
    DEFAULT_GEOCODING_CSV,
    DEFAULT_PROBABILIDADES_CSV,
    UbicacionHistorica,
    cargar_cantidades_pedidos_turnos,
    cargar_y_analizar_historico,
    generar_pedidos_para_un_dia,
)

MAX_PEDIDOS_V2 = 35
MODELS_DIR = PYTHON_ROOT / "models" / "rl"


def _sha256_archivo(ruta: Path) -> str:
    h = sha256()
    with ruta.open("rb") as f:
        for bloque in iter(lambda: f.read(1024 * 1024), b""):
            h.update(bloque)
    return h.hexdigest()


def respaldar_modelo_v1() -> tuple[Path, Path]:
    """
    Conserva una copia intacta del modelo original en `rl_policy_v1.zip`
    y su manifiesto en `rl_policies_v1.json`.
    """
    ruta_zip_actual = MODELS_DIR / "rl_policy.zip"
    ruta_json_actual = MODELS_DIR / "rl_policies.json"
    ruta_zip_v1 = MODELS_DIR / "rl_policy_v1.zip"
    ruta_json_v1 = MODELS_DIR / "rl_policies_v1.json"

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
) -> InstanciaTurno:
    """Genera una InstanciaTurno realista con entre `rango_tareas[0]` y `rango_tareas[1]` tareas."""
    rng = random.Random(seed_escenario)
    turno_str = "MANANA" if rng.random() < 0.55 else "TARDE"
    turno_enum = Turno.MANANA if turno_str == "MANANA" else Turno.TARDE
    inicio_min, fin_obj_min, fin_tol_min = (
        (450, 720, 735) if turno_enum == Turno.MANANA else (840, 1020, 1035)
    )

    tareas_objetivo = rng.randint(rango_tareas[0], rango_tareas[1])
    pedidos_gen = generar_pedidos_para_un_dia(
        ubicaciones=ubicaciones,
        cantidad_pedidos=tareas_objetivo,
        turno=turno_str,
        dia_numero=1,
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
    tareas = tareas[:MAX_PEDIDOS_V2]

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
    ) -> None:
        super().__init__()
        self.ubicaciones = ubicaciones
        self.rango_tareas = rango_tareas
        self.rng = random.Random(seed_base)
        self.max_pedidos = max_pedidos
        self.config_plan = ConfiguracionPlanificacion()
        self.proveedor = ProveedorHaversineAjustado()
        self.config_pol = ConfiguracionPoliticaRL(
            usar_mascara_temporal_dura=True,
            ordenar_slots_canonicamente=True,
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
        )
        return EntornoPlanificacionRL(
            instancia=instancia,
            configuracion=self.config_plan,
            proveedor_viaje=self.proveedor,
            max_pedidos=self.max_pedidos,
            configuracion_temporal=self.config_pol,
        )

    def action_masks(self) -> np.ndarray:
        return self._env_interno.action_masks()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.rng.seed(seed)
        self._env_interno = self._crear_env_nuevo()
        return self._env_interno.reset(seed=seed, options=options)

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        return self._env_interno.step(action)


def transferir_pesos_v1_a_v2(
    ruta_v1: Path,
    modelo_v2: MaskablePPO,
) -> None:
    """
    Transfiere el conocimiento aprendido en `rl_policy_v1.zip` (30 slots, entrenado en slots 0..11)
    hacia la nueva red de 35 slots (`modelo_v2`), propagando los bloques de 23 features por slot
    de los slots entrenados (4..11) hacia los slots extendidos (12..34) para que la red reconozca
    inmediatamente el arrepentimiento local y las holguras en turnos de hasta 35 tareas.
    """
    _asegurar_compatibilidad_numpy()
    planner_v1 = PlanificadorPoliticaRL(model_path=ruta_v1, max_pedidos=30)
    sd_v1 = planner_v1.model.policy.state_dict()
    sd_v2 = modelo_v2.policy.state_dict()

    f_por_ped = EntornoPlanificacionRL.FEATURES_POR_PEDIDO  # 23
    f_glob = EntornoPlanificacionRL.FEATURES_GLOBALES       # 12

    with torch.no_grad():
        # Copiar capas intermedias y de valor que tienen exactamente el mismo tamaño [256, 256] y [1, 256]
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

        # Para la primera capa densa (entrada de observación: 702 -> 817):
        for clave_w in (
            "mlp_extractor.policy_net.0.weight",
            "mlp_extractor.value_net.0.weight",
        ):
            w_old = sd_v1[clave_w]  # [256, 702]
            w_new = sd_v2[clave_w]  # [256, 817]

            # 1. Slots 0..11 (entrenados sólidamente en V1)
            w_new[:, : 12 * f_por_ped].copy_(w_old[:, : 12 * f_por_ped])

            # 2. Bloques plantilla de slots 4..11 para inicializar slots 12..34 con simetría por slot
            plantilla_bloques = [
                w_old[:, s * f_por_ped : (s + 1) * f_por_ped]
                for s in range(4, 12)
            ]
            bloque_promedio = torch.stack(plantilla_bloques, dim=0).mean(dim=0)

            for s_nuevo in range(12, MAX_PEDIDOS_V2):
                ref_bloque = plantilla_bloques[s_nuevo % len(plantilla_bloques)]
                mezcla = 0.70 * bloque_promedio + 0.30 * ref_bloque
                # Escalar por 12/35 para mantener la varianza de activación estable cuando hay 25-35 pedidos activos
                w_new[:, s_nuevo * f_por_ped : (s_nuevo + 1) * f_por_ped].copy_(
                    mezcla * 0.65
                )

            # 3. Features globales (últimas 12 columnas)
            w_new[:, MAX_PEDIDOS_V2 * f_por_ped : MAX_PEDIDOS_V2 * f_por_ped + f_glob].copy_(
                w_old[:, 30 * f_por_ped : 30 * f_por_ped + f_glob]
            )

        # Cabezal de acciones (30 -> 35)
        act_w_old = sd_v1["action_net.weight"]  # [30, 256]
        act_b_old = sd_v1["action_net.bias"]    # [30]
        act_w_new = sd_v2["action_net.weight"]  # [35, 256]
        act_b_new = sd_v2["action_net.bias"]    # [35]

        act_w_new[:12, :].copy_(act_w_old[:12, :])
        act_b_new[:12].copy_(act_b_old[:12])

        filas_plantilla = [act_w_old[s, :] for s in range(4, 12)]
        fila_media = torch.stack(filas_plantilla, dim=0).mean(dim=0)
        bias_medio = act_b_old[4:12].mean()

        for s_nuevo in range(12, MAX_PEDIDOS_V2):
            ref_fila = filas_plantilla[s_nuevo % len(filas_plantilla)]
            act_w_new[s_nuevo, :].copy_(0.70 * fila_media + 0.30 * ref_fila)
            act_b_new[s_nuevo].copy_(bias_medio)

    modelo_v2.policy.load_state_dict(sd_v2)


def evaluar_lote_comparativo(
    ubicaciones: list[UbicacionHistorica],
    ruta_manifiesto_v1: Path,
    ruta_manifiesto_v2: Path,
) -> dict[str, Any]:
    """
    Compara RL_V1 vs RL_V2 vs GREEDY en:
    1. Turnos chicos/medianos (6 a 12 tareas, donde RL_V1 puede ejecutar).
    2. Turnos grandes realistas (13 a 32 tareas, donde sólo RL_V2 puede ejecutar).
    """
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

    semillas_chicas = [500_001 + i * 17 for i in range(20)]
    res_chicos = {
        "RL_V1": {"tardios": 0, "tardanza_min": 0.0, "costo_promedio": 0.0, "exitos": 0},
        "RL_V2": {"tardios": 0, "tardanza_min": 0.0, "costo_promedio": 0.0, "exitos": 0},
        "GREEDY": {"tardios": 0, "tardanza_min": 0.0, "costo_promedio": 0.0, "exitos": 0},
    }

    for s in semillas_chicas:
        inst = construir_instancia_entrenamiento(ubicaciones, (6, 12), s)
        matriz = construir_matriz_viaje(inst, config, proveedor)

        for nombre, fn in (
            ("RL_V1", planner_v1.generar_plan),
            ("RL_V2", planner_v2.generar_plan),
            ("GREEDY", lambda x: generar_plan_greedy(x, configuracion=config, proveedor_viaje=proveedor)),
        ):
            plan = fn(inst)
            est = evaluar_plan_estimado(inst, plan, matriz, config)
            res_chicos[nombre]["tardios"] += int(est.pedidos_tardios)
            res_chicos[nombre]["tardanza_min"] += float(est.tardanza_total_min)
            res_chicos[nombre]["costo_promedio"] += float(est.costo_total) / len(semillas_chicas)
            res_chicos[nombre]["exitos"] += 1

    semillas_grandes = [700_001 + i * 29 for i in range(20)]
    res_grandes = {
        "RL_V1": {"ejecutables": 0, "fallos_por_tope_12": 0},
        "RL_V2": {"ejecutables": 0, "tardios": 0, "tardanza_min": 0.0, "costo_promedio": 0.0},
        "GREEDY": {"ejecutables": 0, "tardios": 0, "tardanza_min": 0.0, "costo_promedio": 0.0},
    }

    for s in semillas_grandes:
        inst = construir_instancia_entrenamiento(ubicaciones, (14, 32), s)
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

        plan_gr = generar_plan_greedy(inst, configuracion=config, proveedor_viaje=proveedor)
        est_gr = evaluar_plan_estimado(inst, plan_gr, matriz, config)
        res_grandes["GREEDY"]["ejecutables"] += 1
        res_grandes["GREEDY"]["tardios"] += int(est_gr.pedidos_tardios)
        res_grandes["GREEDY"]["tardanza_min"] += float(est_gr.tardanza_total_min)
        res_grandes["GREEDY"]["costo_promedio"] += float(est_gr.costo_total) / len(semillas_grandes)

    return {
        "turnos_6_a_12_tareas": res_chicos,
        "turnos_14_a_32_tareas": res_grandes,
    }


def ejecutar_entrenamiento_v2() -> dict[str, Any]:
    ruta_zip_v1, ruta_json_v1 = respaldar_modelo_v1()
    print(f"[OK] Modelo V1 respaldado en: {ruta_zip_v1.name} y {ruta_json_v1.name}")

    ubicaciones = cargar_y_analizar_historico(
        DEFAULT_GEOCODING_CSV,
        ruta_probabilidades=DEFAULT_PROBABILIDADES_CSV,
    )

    env_stage1 = EntornoCurricularMultiInstancia(
        ubicaciones=ubicaciones,
        rango_tareas=(6, 14),
        seed_base=101,
        max_pedidos=MAX_PEDIDOS_V2,
    )

    modelo_v2 = MaskablePPO(
        "MlpPolicy",
        env_stage1,
        learning_rate=1.5e-5,
        n_steps=256,
        batch_size=128,
        n_epochs=6,
        gamma=1.0,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.008,
        policy_kwargs={"net_arch": dict(pi=[256, 256], vf=[256, 256])},
        seed=20261030,
        verbose=0,
    )

    transferir_pesos_v1_a_v2(ruta_zip_v1, modelo_v2)
    print("[OK] Pesos base de RL_V1 transferidos y extendidos a 35 slots.")

    # Etapa 1: Estabilización en 6..14 tareas con ordenamiento canónico
    print("-> Entrenando Etapa 1 (6..14 tareas)...")
    modelo_v2.learn(total_timesteps=2048, reset_num_timesteps=True)

    # Etapa 2: Escalamiento a rango medio-alto (12..24 tareas)
    print("-> Entrenando Etapa 2 (12..24 tareas)...")
    env_stage2 = EntornoCurricularMultiInstancia(
        ubicaciones=ubicaciones,
        rango_tareas=(12, 24),
        seed_base=202,
        max_pedidos=MAX_PEDIDOS_V2,
    )
    modelo_v2.set_env(env_stage2)
    modelo_v2.learn(total_timesteps=2048, reset_num_timesteps=False)

    # Etapa 3: Consolidación en turnos completos hasta 35 tareas (8..32 tareas)
    print("-> Entrenando Etapa 3 (8..32 tareas)...")
    env_stage3 = EntornoCurricularMultiInstancia(
        ubicaciones=ubicaciones,
        rango_tareas=(8, 32),
        seed_base=303,
        max_pedidos=MAX_PEDIDOS_V2,
    )
    modelo_v2.set_env(env_stage3)
    modelo_v2.learn(total_timesteps=2048, reset_num_timesteps=False)

    ruta_zip_v2 = MODELS_DIR / "rl_policy_v2.zip"
    ruta_zip_activo = MODELS_DIR / "rl_policy.zip"
    modelo_v2.save(str(ruta_zip_v2))
    shutil.copy2(ruta_zip_v2, ruta_zip_activo)

    hash_v2 = _sha256_archivo(ruta_zip_activo)

    manifiesto_v2: dict[str, Any] = {
        "version": VERSION_RUNTIME,
        "modelo": "rl_policy.zip",
        "modelo_respaldo_v1": "rl_policy_v1.zip",
        "sha256_modelo": hash_v2,
        "max_pedidos_rl_validado": MAX_PEDIDOS_V2,
        "usar_mascara_temporal_dura": True,
        "ordenar_slots_canonicamente": True,
        "num_rollouts": 4,
    }

    ruta_json_v2 = MODELS_DIR / "rl_policies_v2.json"
    ruta_json_activo = MODELS_DIR / "rl_policies.json"

    manifiesto_v2_copia: dict[str, Any] = dict(manifiesto_v2)
    manifiesto_v2_copia["modelo"] = "rl_policy_v2.zip"
    ruta_json_v2.write_text(
        json.dumps(manifiesto_v2_copia, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    comparacion = evaluar_lote_comparativo(
        ubicaciones=ubicaciones,
        ruta_manifiesto_v1=ruta_json_v1,
        ruta_manifiesto_v2=ruta_json_v2,
    )
    manifiesto_v2["validacion_comparativa_v1_vs_v2"] = comparacion

    ruta_json_v2.write_text(
        json.dumps(
            {**manifiesto_v2_copia, "validacion_comparativa_v1_vs_v2": comparacion},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    ruta_json_activo.write_text(
        json.dumps(manifiesto_v2, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("[OK] Nuevo modelo RL guardado y activado en rl_policy.zip y rl_policies.json")
    print(json.dumps(comparacion, ensure_ascii=False, indent=2))
    return comparacion


if __name__ == "__main__":
    ejecutar_entrenamiento_v2()
