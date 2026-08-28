# PedemonteDigitalTwin_v2

Gemelo digital para la planificación y simulación de repartos de Pedemonte.

## Arquitectura operativa

El modelo AnyLogic importa pedidos desde Excel, construye la instancia de
planificación y delega el plan a `planner.integration.selector_bridge`.

Modos disponibles:

- `RL`: política RL única con máscara temporal dura.
- `HIBRIDO`: RL genera la semilla y GA intenta mejorarla.
- `GA`, `GREEDY` y `RANDOM`: algoritmos complementarios.

El híbrido no reemplaza RL con Greedy. Si la política RL falla, el híbrido
informa el error.

## Estructura principal

- `anylogic/PedemonteDigitalTwin_v2`: simulación e interfaz.
- `python/planner/algorithms`: algoritmos de decisión.
- `python/planner/core`: contratos y estructuras compartidas.
- `python/planner/domain`: validación y preprocesamiento.
- `python/planner/integration`: integración AnyLogic–Python e importación Excel.
- `python/planner/rl`: inferencia y entrenamiento de la política actual.
- `python/planner/routing`: tiempos, costos, ventanas y caché vial.
- `python/models/rl`: manifiesto y checkpoint productivo.
- `python/data/processed/demanda_geografica.csv`: dataset usado por el
  entrenamiento actual.
- `python/data/routing/cache_vial.csv`: matriz vial productiva.
- `python/scripts/training/train_rl_policy.py`: entrenamiento reproducible.
- `python/tests`: pruebas de regresión mantenidas.

## Política RL activa

```text
python/models/rl/rl_policies.json
python/models/rl/rl_policy.zip
```

La inferencia usa máscara temporal dura y está validada hasta 12 pedidos.

## Entrenamiento

El único punto de entrada versionado para entrenar la política actual es:

```text
python/scripts/training/train_rl_policy.py
```

Los checkpoints intermedios y resultados se guardan fuera del control de
versiones según `.gitignore`.

## Validación Python

Desde `python`:

```powershell
$py = (Resolve-Path "..\.venv\Scripts\python.exe").Path

& $py -m compileall -q planner scripts tests
& $py -m pytest -q --tb=short
```

## Comparación para el informe

La comparación experimental final se construirá sobre el selector productivo
actual. No se conserva la infraestructura histórica de contratos y suites de
fases anteriores.
