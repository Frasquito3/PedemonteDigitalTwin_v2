# PedemonteDigitalTwin_v2

Gemelo digital para la planificación y simulación de repartos de Pedemonte.

## Arquitectura operativa

AnyLogic importa los pedidos, construye la instancia y solicita el plan a
Python mediante `planner.integration.selector_bridge`.

Modos disponibles:

- `RL`: política RL única con máscara temporal dura.
- `HIBRIDO`: la política RL genera la semilla y GA intenta mejorarla.
- `GA`, `GREEDY` y `RANDOM`: métodos complementarios.

El Híbrido no sustituye RL con Greedy. Si la política RL falla, el error se
informa explícitamente.

## Estructura de Python

- `planner/algorithms`: Greedy, Random, GA e Híbrido RL→GA.
- `planner/core`: contratos, configuración y estructuras compartidas.
- `planner/domain`: split y validación de instancias y planes.
- `planner/integration`: importación Excel y puentes AnyLogic–Python.
- `planner/routing`: matriz vial, operaciones estimadas y función objetivo.
- `planner/rl/policy_*`: inferencia de la política productiva.
- `planner/rl/training_*`: datos, generación, entorno y validación del
  entrenamiento actual.
- `scripts/training/train_rl_policy.py`: único punto de entrada versionado
  para continuar o reiniciar el entrenamiento.
- `tests`: suite esencial de regresión.

## Política RL activa

```text
python/models/rl/rl_policies.json
python/models/rl/rl_policy.zip
```

La política está validada hasta 12 pedidos y utiliza máscara temporal dura.

## Entrenamiento

El entrenamiento continúa por defecto desde el checkpoint productivo actual:

```powershell
$py = (Resolve-Path "..\.venv\Scripts\python.exe").Path

& $py scripts\training\train_rl_policy.py `
    --run-name policy_training
```

Para comenzar desde cero debe indicarse explícitamente:

```powershell
& $py scripts\training\train_rl_policy.py `
    --from-scratch `
    --run-name policy_training_from_scratch
```

Los resultados se guardan en `python/rl_artifacts` y nunca reemplazan
automáticamente el checkpoint productivo. Toda promoción requiere una
validación y una decisión técnica separadas.

## Validación

Desde `python`:

```powershell
$py = (Resolve-Path "..\.venv\Scripts\python.exe").Path

& $py -m compileall -q planner scripts tests
& $py -m pytest -q --tb=short
```

Resultado esperado de la suite esencial:

```text
76 passed
```

Los identificadores de versión que permanecen dentro de manifiestos,
protocolos o metadatos describen formatos técnicos. No se utilizan versiones
de desarrollo en nombres de archivos.
