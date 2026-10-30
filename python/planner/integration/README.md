# Módulo `planner.integration` — Interoperabilidad con AnyLogic y Excel

El submódulo `integration` contiene los conectores, puentes de comunicación y códecs que permiten a AnyLogic comunicarse bidireccionalmente con los planificadores de Python mediante Pypeline, importar planillas Excel y ejecutar simulaciones discretas vía Alpyne.

---

## Archivos del Módulo

### Puentes de Comunicación con AnyLogic (Pypeline)
- `selector_bridge.py`: Mantiene el selector activo en memoria, expone llamadas de planificación para los 5 algoritmos y devuelve planes en vectores planos para Java.
- `order_import_bridge.py`: Decodifica cadenas URL-encoded (`PEDXLSX1`, `PEDXLSX2`) originadas en AnyLogic para alimentar el importador de pedidos.
- `route_excel_bridge.py`: Reconstruye planes desde vectores numéricos de AnyLogic para archivarlos en Excel sin recalcular.

### Exportación e Importación de Planillas
- `order_excel_import.py`: Lectura tolerante de planillas Excel de pedidos con nombres flexibles de columnas y validaciones de formato.
- `route_excel_export.py`: Generación de planillas Excel estilizadas para los choferes (hojas por camión, horarios y remitos).
- `route_excel_archive.py`: Gestión del versionado y guardado en la carpeta `exports/` del proyecto.

### Simulación y Códecs Alpyne
- `alpyne_codec.py`: Serialización y deserialización bidireccional entre `InstanciaTurno` y vectores nativos de AnyLogic.
- `simulated_execution.py`: Ejecución limpia y headless del modelo AnyLogic exportado mediante Alpyne en procesos Java independientes.
- `simulated_comparison.py`: Runner de experimentación masiva que evalúa y contrasta los 5 algoritmos bajo simulación de eventos discretos.
- `planner_selector.py`: Despachador dinámico que instancia bajo demanda modelos RL, híbridos o metaheurísticas clásicas.

