# PedemonteDigitalTwin_v2

Gemelo digital para la planificación y simulación de repartos de Pedemonte.

## Arquitectura operativa

El modelo AnyLogic importa pedidos desde Excel, construye la instancia de
planificación y delega la generación del plan a Python mediante
`planner.integration.selector_bridge`.

Modos disponibles:

- `RL`: selección pura entre las políticas RL habilitadas.
- `HIBRIDO`: semilla RL obligatoria y refinamiento posterior mediante GA.
- `GA`, `GREEDY` y `RANDOM`: algoritmos complementarios de comparación.

El híbrido no utiliza Greedy como sustituto de RL. Si RL falla, el híbrido
informa un error explícito.

## Estructura principal

- `anylogic/PedemonteDigitalTwin_v2`: modelo de simulación y presentación.
- `python/planner`: lógica de planificación, integración y evaluación.
- `python/models/rl`: manifiesto y checkpoints utilizados en producción.
- `python/data`: demanda geográfica y caché vial.
- `python/scripts/training`: entrenamiento reproducible de las políticas.
- `python/scripts/evaluation`: comparación y validación para el informe.
- `python/tests`: regresión automatizada.
- `python/templates`: plantillas de importación de pedidos.

## Modelos activos

El manifiesto operativo es:

```text
python/models/rl/rl_policies.json
```

Checkpoints:

```text
python/models/rl/rl_policy_balanced.zip
python/models/rl/rl_policy_high_demand.zip
```

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

Resultado de referencia de la limpieza técnica:

```text
210 passed, 6 subtests passed
```

## Artefactos locales

Los resultados, cachés, corridas, checkpoints de entrenamiento y archivos
subidos desde la interfaz no se versionan. Las reglas correspondientes se
encuentran en `.gitignore`.
