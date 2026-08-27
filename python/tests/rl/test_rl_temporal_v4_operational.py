from __future__ import annotations

from hashlib import sha256
from json import dumps
from pathlib import Path
from types import SimpleNamespace

import pytest

from planner.rl.rl_temporal_v4_operational import (
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


def _config(tmp_path: Path) -> ConfiguracionOperacionTemporalV4:
    return ConfiguracionOperacionTemporalV4(
        version=VERSION_OPERACIONAL,
        modelo_extension=tmp_path / "extension.zip",
        modelo_full=tmp_path / "full.zip",
        sha256_extension="",
        sha256_full="",
        max_pedidos_rl_validado=12,
        usar_guardia_greedy=True,
        preferencia_empate_hasta_10="EXTENSION",
        preferencia_empate_desde_11="FULL",
    )


def _preparar(monkeypatch, metricas):
    monkeypatch.setattr(
        "planner.rl.rl_temporal_v4_operational.construir_matriz_viaje",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        "planner.rl.rl_temporal_v4_operational.validar_plan",
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
        "planner.rl.rl_temporal_v4_operational.evaluar_plan_estimado",
        evaluar,
    )


def _instancia(cantidad: int):
    return SimpleNamespace(
        pedidos=[object() for _ in range(cantidad)],
        instancia_id="TEST",
    )


def test_prioriza_menos_pedidos_tardios(monkeypatch, tmp_path):
    extension = _plan("extension")
    full = _plan("full")
    greedy = _plan("greedy")
    _preparar(
        monkeypatch,
        {
            "extension": (1, 1.0, 10.0),
            "full": (0, 0.0, 1000.0),
            "greedy": (1, 0.1, 1.0),
        },
    )
    planner = RLTemporalV4OperationalPlanner(
        configuracion_operacional=_config(tmp_path),
        planner_extension=_PlannerFalso(extension),
        planner_full=_PlannerFalso(full),
        greedy_factory=lambda *_args, **_kwargs: greedy,
    )
    resultado = planner.generar_plan(_instancia(10))
    assert resultado is full
    assert planner.ultima_decision.fuente_seleccionada == "FULL"


def test_en_empate_hasta_10_prefiere_extension(monkeypatch, tmp_path):
    extension = _plan("extension")
    full = _plan("full")
    greedy = _plan("greedy")
    _preparar(
        monkeypatch,
        {nombre: (0, 0.0, 10.0) for nombre in ("extension", "full", "greedy")},
    )
    planner = RLTemporalV4OperationalPlanner(
        configuracion_operacional=_config(tmp_path),
        planner_extension=_PlannerFalso(extension),
        planner_full=_PlannerFalso(full),
        greedy_factory=lambda *_args, **_kwargs: greedy,
    )
    assert planner.generar_plan(_instancia(10)) is extension


def test_en_empate_desde_11_prefiere_full(monkeypatch, tmp_path):
    extension = _plan("extension")
    full = _plan("full")
    greedy = _plan("greedy")
    _preparar(
        monkeypatch,
        {nombre: (0, 0.0, 10.0) for nombre in ("extension", "full", "greedy")},
    )
    planner = RLTemporalV4OperationalPlanner(
        configuracion_operacional=_config(tmp_path),
        planner_extension=_PlannerFalso(extension),
        planner_full=_PlannerFalso(full),
        greedy_factory=lambda *_args, **_kwargs: greedy,
    )
    assert planner.generar_plan(_instancia(11)) is full


def test_greedy_cubre_error_de_los_dos_rl(monkeypatch, tmp_path):
    greedy = _plan("greedy")
    _preparar(monkeypatch, {"greedy": (0, 0.0, 10.0)})
    planner = RLTemporalV4OperationalPlanner(
        configuracion_operacional=_config(tmp_path),
        planner_extension=_PlannerFalso(error=RuntimeError("ext")),
        planner_full=_PlannerFalso(error=RuntimeError("full")),
        greedy_factory=lambda *_args, **_kwargs: greedy,
    )
    assert planner.generar_plan(_instancia(9)) is greedy
    assert set(planner.ultima_decision.errores_por_fuente) == {"EXTENSION", "FULL"}


def test_fuera_de_rango_validado_usa_solo_greedy(monkeypatch, tmp_path):
    greedy = _plan("greedy")
    extension = _PlannerFalso(_plan("extension"))
    full = _PlannerFalso(_plan("full"))
    _preparar(monkeypatch, {"greedy": (0, 0.0, 10.0)})
    planner = RLTemporalV4OperationalPlanner(
        configuracion_operacional=_config(tmp_path),
        planner_extension=extension,
        planner_full=full,
        greedy_factory=lambda *_args, **_kwargs: greedy,
    )
    assert planner.generar_plan(_instancia(13)) is greedy
    assert extension.llamadas == 0
    assert full.llamadas == 0


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
                "usar_guardia_greedy": True,
                "preferencia_empate_hasta_10": "EXTENSION",
                "preferencia_empate_desde_11": "FULL",
            }
        ),
        encoding="utf-8",
    )
    config = cargar_configuracion_operacional(manifest)
    assert config.modelo_extension == extension.resolve()
    assert config.modelo_full == full.resolve()


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
