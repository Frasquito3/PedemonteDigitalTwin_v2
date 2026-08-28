from __future__ import annotations

from hashlib import sha256
from json import dumps
from pathlib import Path
from types import SimpleNamespace

import pytest

from planner.rl.policy_runtime import (
    ConfiguracionOperacionTemporalV4,
    RLTemporalV4OperationalPlanner,
    VERSION_OPERACIONAL,
    cargar_configuracion_operacional,
)


class _PlannerFalso:
    def __init__(self, plan=None, error: Exception | None = None):
        self.plan = plan
        self.error = error
        self.llamadas = 0

    def generar_plan(self, _instancia):
        self.llamadas += 1
        if self.error is not None:
            raise self.error
        return self.plan


def _plan(nombre: str):
    return SimpleNamespace(
        nombre=nombre,
        costo_estimado=0.0,
        tiempo_computo_ms=0.0,
        warnings=[],
    )


def _config(
    tmp_path: Path,
    fuentes: tuple[str, ...] = ("EXTENSION", "FULL"),
) -> ConfiguracionOperacionTemporalV4:
    return ConfiguracionOperacionTemporalV4(
        version=VERSION_OPERACIONAL,
        modelo_extension=tmp_path / "extension.zip",
        modelo_full=tmp_path / "full.zip",
        sha256_extension="",
        sha256_full="",
        max_pedidos_rl_validado=12,
        fuentes_habilitadas=fuentes,
        preferencia_empate_hasta_10="EXTENSION",
        preferencia_empate_desde_11="FULL",
    )


def _preparar(monkeypatch, metricas):
    monkeypatch.setattr(
        "planner.rl.policy_runtime.construir_matriz_viaje",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        "planner.rl.policy_runtime.validar_plan",
        lambda *_args, **_kwargs: SimpleNamespace(valido=True, errores=[]),
    )

    def evaluar(_instancia, plan, _matriz, _configuracion):
        tardios, tardanza, costo = metricas[plan.nombre]
        return SimpleNamespace(
            pedidos_tardios=tardios,
            tardanza_total_min=tardanza,
            costo_total=costo,
        )

    monkeypatch.setattr(
        "planner.rl.policy_runtime.evaluar_plan_estimado",
        evaluar,
    )


def _instancia(cantidad: int):
    return SimpleNamespace(
        pedidos=[object() for _ in range(cantidad)],
        instancia_id="TEST",
    )


def test_rl_puro_prioriza_menos_pedidos_tardios(monkeypatch, tmp_path):
    extension = _plan("extension")
    full = _plan("full")
    _preparar(
        monkeypatch,
        {
            "extension": (1, 1.0, 10.0),
            "full": (0, 0.0, 1000.0),
        },
    )
    planner = RLTemporalV4OperationalPlanner(
        configuracion_operacional=_config(tmp_path),
        planner_extension=_PlannerFalso(extension),
        planner_full=_PlannerFalso(full),
    )
    resultado = planner.generar_plan(_instancia(10))
    assert resultado is full
    assert planner.ultima_decision.fuente_seleccionada == "FULL"
    assert "GREEDY" not in planner.ultimo_detalle


def test_en_empate_hasta_10_prefiere_extension(monkeypatch, tmp_path):
    extension = _plan("extension")
    full = _plan("full")
    _preparar(
        monkeypatch,
        {"extension": (0, 0.0, 10.0), "full": (0, 0.0, 10.0)},
    )
    planner = RLTemporalV4OperationalPlanner(
        configuracion_operacional=_config(tmp_path),
        planner_extension=_PlannerFalso(extension),
        planner_full=_PlannerFalso(full),
    )
    assert planner.generar_plan(_instancia(10)) is extension


def test_en_empate_desde_11_prefiere_full(monkeypatch, tmp_path):
    extension = _plan("extension")
    full = _plan("full")
    _preparar(
        monkeypatch,
        {"extension": (0, 0.0, 10.0), "full": (0, 0.0, 10.0)},
    )
    planner = RLTemporalV4OperationalPlanner(
        configuracion_operacional=_config(tmp_path),
        planner_extension=_PlannerFalso(extension),
        planner_full=_PlannerFalso(full),
    )
    assert planner.generar_plan(_instancia(11)) is full


def test_si_un_checkpoint_falla_devuelve_el_otro_aunque_sea_malo(
    monkeypatch,
    tmp_path,
):
    full = _plan("full")
    _preparar(monkeypatch, {"full": (4, 300.0, 31000.0)})
    planner = RLTemporalV4OperationalPlanner(
        configuracion_operacional=_config(tmp_path),
        planner_extension=_PlannerFalso(error=RuntimeError("ext")),
        planner_full=_PlannerFalso(full),
    )
    assert planner.generar_plan(_instancia(9)) is full
    assert planner.ultima_decision.metricas_seleccionadas.pedidos_tardios == 4
    assert set(planner.ultima_decision.errores_por_fuente) == {"EXTENSION"}


def test_si_ambos_fallan_informa_error_ejecutable(monkeypatch, tmp_path):
    _preparar(monkeypatch, {})
    planner = RLTemporalV4OperationalPlanner(
        configuracion_operacional=_config(tmp_path),
        planner_extension=_PlannerFalso(error=RuntimeError("ext")),
        planner_full=_PlannerFalso(error=RuntimeError("full")),
    )
    with pytest.raises(RuntimeError, match="plan RL ejecutable"):
        planner.generar_plan(_instancia(9))


def test_fuera_de_rango_tecnico_no_usa_otro_algoritmo(monkeypatch, tmp_path):
    extension = _PlannerFalso(_plan("extension"))
    full = _PlannerFalso(_plan("full"))
    _preparar(monkeypatch, {})
    planner = RLTemporalV4OperationalPlanner(
        configuracion_operacional=_config(tmp_path),
        planner_extension=extension,
        planner_full=full,
    )
    with pytest.raises(RuntimeError, match="MAXIMO_12"):
        planner.generar_plan(_instancia(13))
    assert extension.llamadas == 0
    assert full.llamadas == 0


def test_manifiesto_permite_un_solo_checkpoint(monkeypatch, tmp_path):
    full = _plan("full")
    extension = _PlannerFalso(_plan("extension"))
    full_planner = _PlannerFalso(full)
    _preparar(monkeypatch, {"full": (2, 100.0, 500.0)})
    planner = RLTemporalV4OperationalPlanner(
        configuracion_operacional=_config(tmp_path, ("FULL",)),
        planner_extension=extension,
        planner_full=full_planner,
    )
    assert planner.generar_plan(_instancia(8)) is full
    assert extension.llamadas == 0
    assert full_planner.llamadas == 1


def test_carga_manifiesto_y_valida_hashes(tmp_path):
    extension = tmp_path / "extension.zip"
    full = tmp_path / "full.zip"
    extension.write_bytes(b"extension")
    full.write_bytes(b"full")
    manifest = tmp_path / "operational.json"
    manifest.write_text(
        dumps(
            {
                "version": VERSION_OPERACIONAL,
                "modelo_extension": extension.name,
                "sha256_extension": sha256(b"extension").hexdigest(),
                "modelo_full": full.name,
                "sha256_full": sha256(b"full").hexdigest(),
                "max_pedidos_rl_validado": 12,
                "fuentes_habilitadas": ["EXTENSION", "FULL"],
                "preferencia_empate_hasta_10": "EXTENSION",
                "preferencia_empate_desde_11": "FULL",
            }
        ),
        encoding="utf-8",
    )
    config = cargar_configuracion_operacional(manifest)
    assert config.modelo_extension == extension.resolve()
    assert config.modelo_full == full.resolve()
    assert config.fuentes_habilitadas == ("EXTENSION", "FULL")


def test_rechaza_hash_incorrecto(tmp_path):
    extension = tmp_path / "extension.zip"
    full = tmp_path / "full.zip"
    extension.write_bytes(b"extension")
    full.write_bytes(b"full")
    manifest = tmp_path / "operational.json"
    manifest.write_text(
        dumps(
            {
                "version": VERSION_OPERACIONAL,
                "modelo_extension": extension.name,
                "sha256_extension": "0" * 64,
                "modelo_full": full.name,
                "sha256_full": sha256(b"full").hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="extensión"):
        cargar_configuracion_operacional(manifest)
