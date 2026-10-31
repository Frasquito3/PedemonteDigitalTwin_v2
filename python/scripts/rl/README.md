# Módulo `scripts.rl` — Entrenamiento Reproducible de Modelos RL por Configuración JSON

Este módulo centraliza el entrenamiento de políticas de Aprendizaje por Refuerzo (`MaskablePPO`) guiado por archivos de configuración JSON para garantizar **100% de reproducibilidad**.

---

## Estructura de Modelos (`python/models/rl/`)

- `python/models/rl/rl_policy.zip`: **Único modelo activo en producción** consumido por AnyLogic.
- `python/models/rl/rl_policies.json`: **Manifiesto activo** con el hash SHA-256, configuración operativa y los parámetros exactos con los que fue entrenado.
- `python/models/rl/historial/`: Carpeta con todos los modelos anteriores y experimentales (`rl_policy_v1.zip`, `rl_policy_v2.zip`, etc.) junto a sus respectivos manifiestos JSON.

---

## Cómo Entrenar un Modelo o Correr un Nuevo Experimento

1. **Definir o elegir el archivo JSON de parámetros** en `python/scripts/rl/configs/`:
   - `configs/rl_v2_35slots.json`: Configuración del modelo V2 actual (35 slots, orden canónico, currículum de 3 etapas con transferencia de pesos desde V1).
   - `configs/rl_desde_cero_35slots.json`: Plantilla para entrenar un modelo de 35 slots desde cero sin transferencia previa.

2. **Ejecutar el script único de entrenamiento (`train_rl.py`)**:

```bash
# Entrenar con la configuración de un experimento específico
python python/scripts/rl/train_rl.py --config python/scripts/rl/configs/rl_v2_35slots.json

# Entrenar un experimento nuevo y guardarlo solo en models/rl/historial/ sin pisar el modelo activo de AnyLogic
python python/scripts/rl/train_rl.py --config python/scripts/rl/configs/rl_desde_cero_35slots.json --no-activar
```
