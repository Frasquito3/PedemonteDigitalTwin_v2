# Módulo `scripts.simulation` — Simulación y Benchmarking en AnyLogic

Este submódulo agrupa los scripts ejecutables para correr planes de ruteo en el modelo exportado de AnyLogic mediante Alpyne de forma limpia y desatendida.

---

## Archivos del Módulo

### `run_single_simulated_execution.py`
Ejecuta una corrida individual de simulación:
- Carga un turno de pedidos desde un archivo Excel.
- Genera el plan con el método solicitado (`RL`, `HIBRIDO`, `GREEDY`, `RANDOM` o `GA`).
- Lanza el motor de eventos discretos Java de AnyLogic vía Alpyne.
- Exporta las métricas reales resultantes (makespan, tardanzas, desbalance y combustible) a un archivo JSON.

### `run_simulated_comparison.py`
Orquestador de benchmarking sistemático:
- Permite evaluar múltiples turnos (leídos de un directorio o generados estocásticamente en el acto).
- Ejecuta los 5 algoritmos en cada turno a lo largo de múltiples réplicas (por defecto 3 réplicas).
- Registra y aísla las semillas deterministas de cada algoritmo y escenario.
- Consolida tanto las métricas estimadas como las simuladas en tablas `resumen_resultados.csv` y `resumen_resultados.json` alojadas en carpetas versionadas con formato `seed_<seed_inicial>_<fecha>`.

