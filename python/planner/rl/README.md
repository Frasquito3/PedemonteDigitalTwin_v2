# Módulo de Aprendizaje por Refuerzo (RL)

El submódulo `planner.rl` implementa la formulación de aprendizaje por refuerzo profundo (RL) para el ruteo y secuenciamiento de pedidos de hormigón elaborado y materiales, integrándose con **Gymnasium** y frameworks compatibles como **Stable-Baselines3**.

---

## 1. Arquitectura del Entorno (Gymnasium)

El problema de secuenciamiento se modela como un Proceso de Decisión de Markov (MDP) de horizonte finito:

* **Espacio de Acciones:** Discreto de tamaño `max_pedidos`. Cada acción representa despachar un pedido no asignado. Las restricciones duras (capacidad, incompatibilidades de turno, pedidos ya asignados) se gestionan mediante máscaras de acción booleanas (`action_masks()`).
* **Espacio de Observaciones:** Vector continuo normalizado que describe:
  - Estado del tiempo y disponibilidad operativa del turno.
  - Disponibilidad y capacidad de la flota de camiones.
  - Características del pedido candidato (demanda, ventana horaria, urgencia, tipo de descarga).
  - Estimación anticipada de consecuencias temporales de segundo orden (`policy_estimator.py`).

---

## 2. Modelado de Recompensa y Regret

El módulo `reward.py` define la señal de recompensa orientada a guiar al agente hacia soluciones que superen a las heurísticas voraces clásicas:

* **Ventaja Relativa contra Greedy (`VENTAJA_GREEDY_RELATIVA`):** Evalúa la ganancia o pérdida porcentual frente al costo del paso voraz inmediato.
* **Penalización por Tardanza:** Fuertes penalizaciones cuadráticas o lineales sobre pedidos que exceden su ventana horaria de entrega (`hora_hasta_min`).
* **Arrepentimiento Local (Regret):** Estimación incremental del costo de oportunidad por haber postergado pedidos más críticos.

---

## 3. Pipeline de Entrenamiento y Curriculum

Para asegurar estabilidad en políticas complejas, el entrenamiento utiliza un esquema de curriculum progresivo:

1. **Generador de Instancias (`instance_generator.py`):** Genera escenarios sintéticos o basados en ubicaciones históricas reales (`training_data.py`), garantizando particiones disjuntas `TRAIN`, `VALIDATION` y `TEST` sin fuga física por dirección.
2. **Curriculum Adaptativo (`training_curriculum.py`):** Avanza desde instancias simples de baja demanda hacia instancias de alta congestión con ventanas conflictivas y fraccionamiento (*split*).
3. **Validación Externa (`training_validation.py`):** Evalúa periódicamente checkpoints contra casos canónicos de prueba (`training_cases.py`), seleccionando el modelo que minimice el riesgo de tardanzas operativas.

---

## 4. Inferencia y Producción

En tiempo de ejecución productivo, `PlanificadorOperativoRL` y `PlanificadorPoliticaRL` consumen el modelo entrenado (ej. Maskable PPO) para emitir secuencias de despacho factibles en milisegundos, sirviendo tanto de planificador autónomo como de generador de semilla inicial para el algoritmo genético (`hybrid_rl_ga.py`).

