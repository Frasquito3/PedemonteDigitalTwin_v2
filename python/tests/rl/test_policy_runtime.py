from __future__ import annotations
from hashlib import sha256
from json import dumps
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import pytest
from planner.rl.policy_runtime import ConfiguracionOperacionRL, FUENTE_RL_UNICA, PlanificadorOperativoRL, VERSION_RUNTIME, cargar_configuracion_operacional

class _PlannerFalso:

    def __init__(self, plan=None, error: Exception | None=None):
        self.plan = plan
        self.error = error
        self.llamadas = 0

    def generar_plan(self, _instancia):
        self.llamadas += 1
        if self.error is not None:
            raise self.error
        return self.plan

def _plan(nombre: str='rl'):
    return SimpleNamespace(nombre=nombre, costo_estimado=0.0, tiempo_computo_ms=0.0, warnings=[])

def _config(tmp_path: Path) -> ConfiguracionOperacionRL:
    return ConfiguracionOperacionRL(version=VERSION_RUNTIME, modelo=tmp_path / 'rl_policy.zip', sha256_modelo='', max_pedidos_rl_validado=12, usar_mascara_temporal_dura=True)

def _preparar(monkeypatch, *, tardios=0, tardanza=0.0, costo=10.0):
    monkeypatch.setattr('planner.rl.policy_runtime.construir_matriz_viaje', lambda *_args, **_kwargs: object())
    monkeypatch.setattr('planner.rl.policy_runtime.validar_plan', lambda *_args, **_kwargs: SimpleNamespace(valido=True, errores=[]))
    monkeypatch.setattr('planner.rl.policy_runtime.evaluar_plan_estimado', lambda *_args, **_kwargs: SimpleNamespace(pedidos_tardios=tardios, tardanza_total_min=tardanza, costo_total=costo))

def _instancia(cantidad: int):
    return SimpleNamespace(pedidos=[object() for _ in range(cantidad)], instancia_id='TEST')

def test_ejecuta_una_sola_politica_y_registra_decision(monkeypatch, tmp_path):
    _preparar(monkeypatch, tardios=1, tardanza=4.5, costo=120.0)
    plan = _plan()
    planner_falso = _PlannerFalso(plan)
    planner = PlanificadorOperativoRL(configuracion_operacional=_config(tmp_path), planner_rl=planner_falso)
    assert planner.generar_plan(_instancia(10)) is plan
    assert planner_falso.llamadas == 1
    assert planner.ultima_decision is not None
    assert planner.ultima_decision.fuente_seleccionada == FUENTE_RL_UNICA
    assert planner.ultima_decision.metricas.pedidos_tardios == 1
    assert 'arquitectura=RL_PURO_POLITICA_UNICA' in planner.ultimo_detalle
    assert 'mascara_temporal=DURA' in planner.ultimo_detalle
    assert 'GREEDY' not in planner.ultimo_detalle

def test_fallo_de_politica_informa_error_ejecutable(monkeypatch, tmp_path):
    _preparar(monkeypatch)
    planner = PlanificadorOperativoRL(configuracion_operacional=_config(tmp_path), planner_rl=_PlannerFalso(error=RuntimeError('fallo controlado')))
    with pytest.raises(RuntimeError, match='política RL única no pudo generar'):
        planner.generar_plan(_instancia(9))

def test_fuera_de_rango_tecnico_no_ejecuta_politica(monkeypatch, tmp_path):
    _preparar(monkeypatch)
    planner_falso = _PlannerFalso(_plan())
    planner = PlanificadorOperativoRL(configuracion_operacional=_config(tmp_path), planner_rl=planner_falso)
    with pytest.raises(RuntimeError, match='máximo=12'):
        planner.generar_plan(_instancia(13))
    assert planner_falso.llamadas == 0

def test_carga_manifiesto_unico_y_valida_hash(tmp_path):
    modelo = tmp_path / 'rl_policy.zip'
    modelo.write_bytes(b'single-policy')
    manifest = tmp_path / 'rl_policies.json'
    manifest.write_text(dumps({'version': VERSION_RUNTIME, 'modelo': modelo.name, 'sha256_modelo': sha256(b'single-policy').hexdigest(), 'max_pedidos_rl_validado': 12, 'usar_mascara_temporal_dura': True}), encoding='utf-8')
    config = cargar_configuracion_operacional(manifest)
    assert config.modelo == modelo.resolve()
    assert config.usar_mascara_temporal_dura is True

def test_rechaza_hash_incorrecto(tmp_path):
    modelo = tmp_path / 'rl_policy.zip'
    modelo.write_bytes(b'single-policy')
    manifest = tmp_path / 'rl_policies.json'
    manifest.write_text(dumps({'version': VERSION_RUNTIME, 'modelo': modelo.name, 'sha256_modelo': '0' * 64}), encoding='utf-8')
    with pytest.raises(ValueError, match='SHA-256'):
        cargar_configuracion_operacional(manifest)

def test_construye_planner_con_mascara_dura(monkeypatch, tmp_path):
    modelo = tmp_path / 'rl_policy.zip'
    modelo.write_bytes(b'policy')
    config = ConfiguracionOperacionRL(version=VERSION_RUNTIME, modelo=modelo, sha256_modelo='', max_pedidos_rl_validado=12, usar_mascara_temporal_dura=True)
    capturado = {}

    class PlannerConstruido:

        def __init__(self, **kwargs):
            capturado.update(kwargs)

        def generar_plan(self, _instancia):
            return _plan()
    module = ModuleType('planner.rl.policy_planner')
    module.PlanificadorPoliticaRL = PlannerConstruido
    monkeypatch.setitem(sys.modules, 'planner.rl.policy_planner', module)
    PlanificadorOperativoRL(configuracion_operacional=config)
    temporal = capturado['configuracion_temporal']
    assert temporal.usar_mascara_temporal_dura is True

def test_rechaza_maximo_no_positivo(tmp_path):
    modelo = tmp_path / 'rl_policy.zip'
    modelo.write_bytes(b'single-policy')
    manifest = tmp_path / 'rl_policies.json'
    manifest.write_text(dumps({'version': VERSION_RUNTIME, 'modelo': modelo.name, 'sha256_modelo': sha256(b'single-policy').hexdigest(), 'max_pedidos_rl_validado': 0}), encoding='utf-8')
    with pytest.raises(ValueError, match='debe ser > 0'):
        cargar_configuracion_operacional(manifest)
