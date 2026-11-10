# Módulo `scripts.simulation` - Simulación y Benchmarking en AnyLogic

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


### Persistencia y auditoría del estudio

El ejecutor guarda `manifest_corridas.json` antes de la primera réplica; conserva las
entradas originales en `entradas/`, las representaciones numéricas en `instancias/`,
y los planes exactos de los cinco métodos en `planes/`. Al terminar cada réplica,
actualiza de manera atómica `resumen_resultados.json`, `resumen_resultados.csv` y
`tiempos_computo.json`. Esta estrategia evita perder toda la campaña al fallar
una ejecución tardía. El resumen CSV conserva sus columnas para AnyLogic.

La carpeta `analisis/` contiene `estadisticas_metodos.csv`, `comparaciones_rl.csv`
y `auditoria.json`. Los intervalos de confianza del 95 % se calculan sobre medias
por escenario de las diferencias emparejadas RL - método; las réplicas del mismo
escenario no se consideran observaciones independientes. Son intervalos sin
corrección por comparaciones múltiples, por lo cual no bastan por sí solos para
hacer declaraciones de significación conjunta de todos los algoritmos.

Para reanalizar archivos existentes sin simular:

```powershell
cd python
python -m scripts.simulation.analyze_experiment --directorio "data/results/estudios/.../seed_..."
```

Para recuperar una ejecución interrumpida conservando entradas y semillas:

```powershell
cd python
python -m scripts.simulation.run_simulated_comparison --reanudar-desde "data/results/estudios/.../seed_..."
```

Opcionalmente, agregar `--reintentar-fallidas` para repetir las réplicas con
métodos fallidos. La reanudación se rechaza si cambiaron los hashes del código
ejecutor, el generador, el modelo exportado, la política RL o la caché vial.
Requiere mantener los archivos de la corrida y los componentes de software usados.

`tiempos_computo.json` registra importación, planificación, simulación AnyLogic
y duración de cada réplica, además de los tiempos globales de las sesiones.
El generador Python registra `tiempo_generacion_s` en su metadata. Cuando la
preparación de escenarios y GIS se realiza desde la interfaz gráfica de AnyLogic,
esas fases anteceden al ejecutor; su medición debe integrarse en el modelo Java.

Los resultados experimentales pueden incluir ubicaciones operativas. No publicar
las carpetas completas en GitHub Pages ni en un repositorio público sin revisar
su contenido.
