from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
from planner.integration.planner_selector import SelectorPlanificadores

def test_selector_detecta_manifiesto_json(monkeypatch, tmp_path: Path):
    manifest = tmp_path / 'rl_policies.json'
    manifest.write_text('{}', encoding='utf-8')

    class PlannerFalso:

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def generar_plan(self, _instancia):
            return SimpleNamespace()
    monkeypatch.setattr('planner.rl.policy_runtime.PlanificadorOperativoRL', PlannerFalso)
    selector = SelectorPlanificadores(model_path_rl=manifest)
    planner = selector._obtener_planner_rl()
    assert isinstance(planner, PlannerFalso)
    assert planner.kwargs['manifest_path'] == manifest.resolve()
