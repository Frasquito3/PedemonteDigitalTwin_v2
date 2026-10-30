# Módulo `planner.domain` — Lógica de Negocio y Validaciones

El submódulo `domain` implementa las reglas operativas del corralón, transformando los pedidos originales en tareas unitarias factibles y verificando la integridad de instancias y planes.

---

## Archivos del Módulo

### `preprocess.py`
Procesamiento de pedidos antes de su paso a los algoritmos:
- **Order Split**: Si un pedido supera la capacidad máxima de tolva del camión (8 unidades), se desagrega automáticamente en $K$ sub-pedidos de hasta 8 unidades (`PED_001_P1`, `PED_001_P2`).
- Asignación de ventanas horarias uniformes a las partes derivadas.
- Manejo de requerimientos de descarga por volcador en sub-pedidos individuales.

### `validator.py`
Garantía estricta de factibilidad operativa:
- `validar_instancia`: Verifica límites de coordenadas, coherencia entre horario inicio/fin de turno, existencia de clientes y parámetros válidos de vehículos.
- `validar_plan`: Comprueba que cada viaje no exceda las 8 unidades de tolva, que los pedidos con volcador se descarguen exclusivamente al final del viaje, que no existan solapamientos temporales entre camiones y que todos los pedidos asignados pertenezcan a la instancia.

