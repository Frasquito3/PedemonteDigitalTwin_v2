<div align="center">

# Desarrollo, Decisiones Arquitectónicas y Despliegue

[Inicio](../README.md) •
[Documentación](README.md) •
[Primeros Pasos](getting-started.md) •
[Diseño del Sistema](system-design.md) •
[Manual de Operación](user-manual.md) •
[Calidad y Experimentos](quality-and-experiments.md) •
**Desarrollo y Despliegue**

</div>

---

Este documento registra los fundamentos de las decisiones de diseño adoptadas en **Pedemonte Digital Twin**, el flujo de trabajo para introducir cambios en el código y las consideraciones de despliegue del sistema.

## Decisiones Arquitectónicas Fundamentales

1. **Integración Local AnyLogic Desktop + Python (Pypeline / Alpyne)**
   - *Justificación:* La ejecución en un entorno local unificado elimina la latencia de red durante la inferencia y la validación de planes, simplifica la sincronización del estado entre el simulador de eventos discretos y los optimizadores en Python, y permite ejecutar campañas de benchmarking headless mediante Alpyne sin infraestructura externa.
2. **Máscara Temporal Dura en Aprendizaje por Refuerzo (MaskablePPO)**
   - *Justificación:* En lugar de penalizar acciones inválidas únicamente mediante recompensas negativas (*soft constraints*), el entorno `planner.rl.policy_env` aplica un enmascaramiento estricto sobre el espacio de acciones. Esto garantiza que la política neuronal solo explore transiciones que respeten la capacidad del camión, las ventanas horarias y la regla física de descarga por volcador.
3. **Caché Vial Estricta sin Fallback Silencioso**
   - *Justificación:* Para evitar que variaciones en servicios externos de ruteo (OSRM / OpenStreetMap) alteren los tiempos de viaje entre corridas o contaminen las comparaciones entre algoritmos, toda la planificación productiva consulta exclusivamente la matriz precalculada en `python/data/routing/cache_vial.csv` con `permitir_fallback_vial=False`.
4. **Persistencia Atómica en Pipelines de Datos**
   - *Justificación:* Los procesos de extracción OCR y geocodificación operan sobre lotes extensos sujetos a límites de cuota o interrupciones de red. La escritura mediante archivos temporales en el mismo volumen seguida de reemplazo atómico (`os.replace`) previene la corrupción de los archivos de estado (`progreso.json`, `direcciones.json` y CSVs geolocalizados).

---

<a id="flujo-de-desarrollo-y-calidad-de-codigo"></a>
## Flujo de Desarrollo y Calidad de Código

Para mantener la estabilidad del repositorio al incorporar mejoras o corregir defectos, se aplica el siguiente flujo de validación:

1. **Formateo y análisis estático con Ruff (`pyproject.toml`):**
   - Los hooks en `.githooks/pre-commit` y `.githooks/pre-push` ejecutan `ruff check` y `ruff format` sobre los archivos Python modificados.
2. **Verificación de tipos con Pyrefly (`python/pyrefly.toml`):**
   - Todo módulo, script o prueba debe conservar compatibilidad de tipos bajo Python 3.11 (`pyrefly check` sin errores ni advertencias).
3. **Integración Continua (`.github/workflows/ci.yml`):**
   - Cada `push` o `pull_request` hacia `main` ejecuta tres trabajos independientes en GitHub Actions (`ubuntu-latest`, Python 3.11):
     - `lint`: `python -m ruff check python` y `python -m ruff format --check python`.
     - `test`: `python -m pytest -q --tb=short`.
     - `typecheck`: `pyrefly check`.
4. **Despliegue Automático de Documentación API (`.github/workflows/docs.yml`):**
   - Los docstrings en formato Google de los paquetes `planner` y `scripts`, junto con los diagramas Mermaid y la plantilla personalizada en `docs/pdoc_template/`, se compilan mediante `pdoc` y se publican automáticamente en GitHub Pages.

---

<a id="restricciones-de-despliegue-anylogic-cloud"></a>
## Restricciones de Despliegue: AnyLogic Cloud

> [!CAUTION]
> La arquitectura actual está diseñada para **AnyLogic Desktop** y **ejecución headless local con Alpyne**; no es directamente ejecutable dentro del entorno sandboxed de **AnyLogic Cloud**.

**Motivos técnicos:**
- El conector Pypeline invoca un intérprete local de Python (`python.exe` / `python3`) con acceso directo al sistema de archivos (`python/models/rl/`, `python/data/routing/cache_vial.csv`) y librerías nativas (`numpy`, `torch`, `stable-baselines3`).
- El entorno de ejecución compartido de AnyLogic Cloud restringe la creación de subprocesos locales arbitrarios y no dispone del entorno virtual Python ni de los artefactos del repositorio en disco.

### Arquitectura Requerida para un Eventual Despliegue en Nube
En caso de requerirse en el futuro una versión compatible con AnyLogic Cloud, la integración deberá migrar de un esquema IPC local (Pypeline) a una arquitectura cliente-servidor:
1. Exponer el paquete `planner` a través de un servicio HTTP/REST (por ejemplo, mediante FastAPI) desplegado en un contenedor independiente con los modelos RL y la caché vial embebidos.
2. Reemplazar las llamadas a `pyCommunicator` en AnyLogic por peticiones HTTP hacia dicho servicio.

---

## Líneas de Evolución Futura

Los siguientes ejes constituyen extensiones posibles sobre la base actual del proyecto:

- **Servicio REST Desacoplado:** Encapsular `selector_bridge` en una API web liviana para permitir planificación remota desde clientes externos o entornos en la nube.
- **Dimensionamiento Dinámico de Flota:** Extender el espacio de observación y acción del entorno RL para admitir un número variable de camiones activos (por ejemplo, evaluación de incorporación de un tercer camión en días de demanda pico).
- **Actualización Automatizada de Matriz Vial:** Incorporar un utilitario que detecte coordenadas nuevas al importar una planilla, consulte las distancias reales sobre la red vial de OpenStreetMap/OSRM y actualice de forma controlada y versionada `cache_vial.csv`.
