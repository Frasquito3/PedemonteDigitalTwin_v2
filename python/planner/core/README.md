# Módulo `planner.core` — Modelos de Datos y Protocolos Base

El submódulo `core` centraliza las definiciones estructurales, los esquemas tipados de entrada y salida, la configuración global del gemelo digital y los protocolos de interfaz comunes a todos los algoritmos.

---

## Archivos del Módulo

### `schema.py`
Define los tipos inmutables y estructuras de datos del dominio:
- `InstanciaTurno`: Representación completa del turno (pedidos, camiones, coordenadas del corralón, horarios y semillas).
- `PedidoInput`: Atributos de cada entrega (coordenadas, unidades de tolva, ventana horaria solicitada, volcador).
- `PlanTurno`: Solución completa producida por un planificador, compuesta por viajes asignados a cada vehículo.
- `ViajeCamion`: Secuencia ordenada de paradas con horas estimadas de llegada, salida y regreso a la planta.
- Enums operativos: `Turno` (MANANA, TARDE), `EstadoPedido`, `AlgoritmoPlanificacion`, `TipoActividadCamion`.

### `config.py`
Configuración de ponderaciones y costos de la función objetivo:
- `ConfiguracionPlanificacion`: Coeficientes de combustible, costo por kilómetro, penalizaciones por tardanza de entrega y penalización por desbalance horario entre camiones.

### `base.py`
Contratos e interfaces:
- `PlanificadorTurno`: Protocolo tipado que define el método `generar_plan(instancia: InstanciaTurno) -> PlanTurno`, asegurando que cualquier algoritmo nuevo sea intercambiable en el selector.

