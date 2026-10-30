# Paquete `planner` — Sistema de Planificación Logística

El paquete `planner` constituye el núcleo computacional de inteligencia y optimización combinatoria del **Gemelo Digital Pedemonte**. Su objetivo es recibir los pedidos y restricciones de un turno de trabajo, validarlos según las reglas del dominio y construir una asignación óptima y factible de viajes para la flota de camiones.

---

## Estructura del Paquete

```
python/planner/
├── core/           # Modelos de datos base, esquemas de turnos y protocolos
├── domain/         # Lógica del negocio, partición de pedidos y validadores
├── routing/        # Decodificación de rutas, función objetivo y matrices viales
├── algorithms/     # Motores de optimización (RL, Híbrido, GA, Greedy, Random)
├── rl/             # Entornos Gymnasium, inferencia neuronal y curriculum
└── integration/    # Puentes Java con AnyLogic (Pypeline), Alpyne y Excel
```

---

## Contenido de las Áreas Principales

### 1. `core` — Núcleo y Esquemas
Define los contratos y estructuras de datos fundamentales que atraviesan todo el sistema:
- `schema.py`: Dataclasses inmutables para `PedidoInput`, `InstanciaTurno`, `PlanTurno`, `ViajeCamion` y estados operativos.
- `config.py`: Parámetros de costos (combustible, penalizaciones por tardanza, desbalance de fin de turno).
- `base.py`: Protocolo canónico `PlanificadorTurno` que deben satisfacer todos los algoritmos.

### 2. `domain` — Dominio y Reglas del Negocio
Aplica las restricciones operativas y sanitiza la entrada:
- `preprocess.py`: Partición de pedidos que exceden la capacidad nominal del camión (*order split*) y asignación de identificadores de parte.
- `validator.py`: Verificación exhaustiva de ventanas horarias, turnos válidos, no negatividad de coordenadas y factibilidad de descarga volcador.

### 3. `routing` — Ruteo y Modelado Espacial
Traduce secuencias de pedidos a itinerarios físicos y calcula su costo:
- `decoder.py`: Mapea permutaciones de pedidos a viajes discretos asignados a camiones respetando capacidad y ventanas.
- `objective.py`: Función de costo multiobjetivo unificada (distancia, duración, tardanzas y desbalance).
- `travel.py`: Proveedores de tiempos y distancias (`ProveedorHaversineAjustado` y matrices precalculadas).
- `vial_cache.py`: Caché de red vial persistente en disco para evitar recalcular rutas idénticas.

### 4. `algorithms` — Algoritmos de Resolución
Implementa los cinco métodos comparables del gemelo digital:
- `greedy.py`: Heurística voraz multicriterio basada en prioridad y ventanas de entrega.
- `random_feasible.py`: Muestreo estocástico uniforme de cota inferior.
- `ga.py`: Algoritmo genético por permutaciones con operadores OX, Swap e Inversión.
- `hybrid_rl_ga.py`: Optimizador híbrido que utiliza la política de RL como semilla y el GA como refinador local.

### 5. `rl` — Aprendizaje por Refuerzo
Provee los modelos neuronales y el entorno de entrenamiento:
- `policy_runtime.py`: `PlanificadorOperativoRL` para inferencia productiva en milisegundos.
- `policy_env.py`: Entorno Gymnasium con máscara temporal dura para impedir selecciones infactibles.
- `policy_config.py`: Hiperparámetros de la política, ordenamiento canónico de slots y rollouts.

### 6. `integration` — Interoperabilidad con AnyLogic y Excel
Conecta el código Python con la simulación interactiva:
- `selector_bridge.py`: Puente global Pypeline invocado desde los eventos Java de AnyLogic.
- `order_excel_import.py` y `route_excel_export.py`: Importación y exportación de planillas de despacho.
- `simulated_comparison.py`: Orquestador de benchmarking masivo en simulación headless vía Alpyne.

