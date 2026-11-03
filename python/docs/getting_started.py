"""Guía de Instalación y Primeros Pasos."""

from __future__ import annotations

from pathlib import Path

_md_path = Path(__file__).resolve().parents[2] / "docs" / "getting-started.md"
if _md_path.is_file():
    _text = _md_path.read_text(encoding="utf-8")
    for _md, _html in [
        ("getting-started.md", "getting_started.html"),
        ("system-design.md", "system_design.html"),
        ("user-manual.md", "user_manual.html"),
        ("quality-and-experiments.md", "quality_and_experiments.html"),
        ("development-and-deployment.md", "development_and_deployment.html"),
        ("README.md", "../index.html"),
        ("../README.md", "../index.html"),
    ]:
        _text = _text.replace(f"({_md})", f"({_html})")
        _text = _text.replace(f'href="{_md}"', f'href="{_html}"')
    __doc__ = _text

__all__: list[str] = []
