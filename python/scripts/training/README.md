# Módulo `scripts.training` — Entrenamiento de Políticas de Aprendizaje por Refuerzo

Este módulo contiene los scripts de entrenamiento de la política RL productiva mediante `MaskablePPO` (sb3-contrib).

---

## Archivos del Módulo

### `train_rl_policy.py`
Ejecuta el pipeline de entrenamiento curricular:
- Configura entornos Gymnasium con máscara temporal dura.
- Implementa selección de modelos basada en un ranking lexicográfico sobre instancias de validación fijas (ej. B04).
- Preserva checkpoints periódicos y genera artefactos de evaluación en `python/rl_artifacts/`.

