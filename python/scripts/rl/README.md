# Módulo `scripts.rl` — Evolución y Actualización de Modelos RL

Este módulo contiene scripts de transferencia de conocimiento y escalamiento dimensional para las políticas neuronales de ruteo.

---

## Archivos del Módulo

### `train_rl_v2.py`
Entrena la versión extendida de la política (RL_V2, hasta 35 tareas):
- Respalda el modelo original en `rl_policy_v1.zip` y `rl_policies_v1.json`.
- Transfiere los pesos aprendidos de 30 a 35 slots mediante simetría y escalamiento de varianza.
- Ejecuta un curriculum en 3 etapas (baja, media y alta complejidad) y actualiza los artefactos productivos en `models/rl/`.

