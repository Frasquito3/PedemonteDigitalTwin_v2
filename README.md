# PedemonteDigitalTwin_v2

Gemelo digital para la planificación y simulación de repartos de Pedemonte.

## Arquitectura operativa

El modelo AnyLogic importa pedidos desde Excel, construye la instancia de
planificación y delega la generación del plan a Python mediante
`planner.integration.selector_bridge`.

Modos disponibles:

- `RL`: una única política RL productiva con máscara temporal dura.
- `HIBRIDO`: la política RL genera la semilla y GA intenta refinarla.
- `GA`, `GREEDY` y `RANDOM`: algoritmos complementarios de comparación.

El híbrido no utiliza Greedy como sustituto de RL. Si RL falla, el híbrido
informa un error explícito.

## Estructura principal

- `anylogic/PedemonteDigitalTwin_v2`: modelo de simulación y presentación.
- `python/planner`: lógica de planificación e integración.
- `python/models/rl`: manifiesto y política RL productiva.
- `python/data`: demanda geográfica y caché vial.
- `python/scripts/training`: entrenamiento reproducible de la política actual.
- `python/scripts/evaluation`: comparación y validación para el informe.
- `python/tests`: regresión automatizada.
- `python/templates`: plantillas de importación de pedidos.

## Política RL activa

Manifiesto:

```text
python/models/rl/rl_policies.json
```

Checkpoint:

```text
python/models/rl/rl_policy.zip
```

La inferencia utiliza máscara temporal dura y está validada para instancias de
hasta 12 pedidos. La decisión de promoción conserva documentado el compromiso
frente al selector anterior de dos checkpoints:

- política única: 231/246 casos sin riesgo, 22 pedidos tardíos y 215,977 min;
- selector anterior: 229/246 casos sin riesgo, 21 pedidos tardíos y 247,066 min.

Se priorizó la política única porque aumenta los casos completamente libres de
tardanza y reduce la tardanza total, aceptando un pedido tardío acumulado
adicional en el holdout.

## Caché vial

```text
python/data/routing/cache_vial.csv
```

La ejecución operativa utiliza la caché en modo estricto.

## Validación Python

Desde la carpeta `python`:

```powershell
$py = (Resolve-Path "..\.venv\Scripts\python.exe").Path

& $py -m compileall -q planner scripts tests
& $py -m pytest -q --tb=short
```

## Artefactos locales

Los resultados, cachés, corridas, checkpoints intermedios de entrenamiento y
archivos subidos desde la interfaz no se versionan. Las reglas correspondientes
se encuentran en `.gitignore`.
