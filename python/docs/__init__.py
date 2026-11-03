"""# Documentación Técnica del Proyecto

Este paquete compila las guías maestras y manuales de arquitectura, operación,
calidad y metodología experimental de **Pedemonte Digital Twin**.

## Guías del Sistema

- [`docs.getting_started`](getting_started.html): Requisitos del sistema, configuración local con Python 3.11 y AnyLogic, verificación de integridad y primera corrida.
- [`docs.system_design`](system_design.html): Arquitectura integral, formulación de los motores de despacho, AnyLogic RL Environment y función objetivo.
- [`docs.user_manual`](user_manual.html): Manual operativo, interfaz web, formatos de importación Excel y resolución de problemas.
- [`docs.quality_and_experiments`](quality_and_experiments.html): Suite de pruebas automatizadas, chequeo estático, auditoría SHA-256 y marco experimental.
- [`docs.development_and_deployment`](development_and_deployment.html): Flujo de contribución, decisiones técnicas, pipelines de CI/CD y despliegue.
"""

from __future__ import annotations

__all__ = [
    "development_and_deployment",
    "getting_started",
    "quality_and_experiments",
    "system_design",
    "user_manual",
]
