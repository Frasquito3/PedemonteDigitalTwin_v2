<div align="center">

# Calidad de Software y Metodología Experimental

[Inicio](../README.md) •
[Documentación](README.md) •
[Primeros Pasos](getting-started.md) •
[Diseño del Sistema](system-design.md) •
[Manual de Operación](user-manual.md) •
**Calidad y Experimentos** •
[Desarrollo y Despliegue](development-and-deployment.md)

</div>

---

La confiabilidad técnica de **Pedemonte Digital Twin** se sustenta en tres pilares: verificación automatizada continua (pruebas unitarias, de integración, tipado estático y formato), trazabilidad criptográfica de los modelos de Aprendizaje por Refuerzo y un protocolo experimental reproducible para contrastar algoritmos.

<a id="suite-de-regresion-y-verificacion-estatica"></a>
## Suite de Regresión y Verificación Estática

Todo cambio en el paquete `python/` es validado automáticamente mediante tres herramientas complementarias configuradas en el flujo de integración continua (`.github/workflows/ci.yml`):

### 1. Pruebas Automatizadas (`pytest`)

```powershell
Set-Location .\python
..\.venv\Scripts\python.exe -m pytest -q --tb=short
```

- **Cobertura actual:** **174 pruebas automatizadas** distribuidas en `tests/algorithms/`, `tests/data_processing/`, `tests/domain/`, `tests/integration/`, `tests/rl/` y `tests/routing/`.
- **Áreas verificadas:** Partición de pedidos (*order split*), validador de restricciones duras, precedencia de volcador, cálculo y auditoría de la función objetivo, concurrencia de empleados en corralón, caché vial estricta, códecs vectoriales Pypeline/Alpyne, importación/exportación Excel, escritura atómica de datos y ejecución de la política RL.

### 2. Verificación Estática de Tipos (`pyrefly`)

```powershell
Set-Location .\python
..\.venv\Scripts\pyrefly.exe check --python-interpreter-path ..\.venv\Scripts\python.exe
```

- Configurado en `python/pyrefly.toml` sobre los directorios `planner`, `tests` y `scripts` con resultado exigido de **0 errores y 0 advertencias**.

### 3. Linting y Formato de Código (`ruff`)

```powershell
.\.venv\Scripts\ruff.exe check python
.\.venv\Scripts\ruff.exe format --check python
```

- Configurado en `pyproject.toml` en la raíz del proyecto (longitud de línea de 100 caracteres, Python 3.11, reglas `E`, `W`, `F`, `I`, `UP`).

---

<a id="checkpoint-rl-y-reproducibilidad"></a>
## Checkpoint RL y Reproducibilidad

El entorno de ejecución (`planner.rl.policy_runtime`) valida en cada inicialización que el modelo cargado corresponda exactamente al artefacto declarado en `python/models/rl/rl_policies.json`:

- **Versión de contrato:** `pedemonte-rl-single-policy-v1`
- **Archivo productivo:** `python/models/rl/rl_policy.zip` (experimento `v3`, entrenado por transferencia de aprendizaje sobre `rl_policy_v2.zip` con semilla global `20261031`).
- **Hash SHA-256 verificado:**
  ```text
  df1edc421763eaa7aa426c596ae16998d5867ddc720eda8d2cf88923f5fe434f
  ```
- **Rango operativo validado (`max_pedidos_rl_validado`):** Hasta **35 tareas operativas por turno** (incluyendo las particiones generadas por *order split*).

> [!IMPORTANT]
> Cualquier modificación en la estructura del vector de observación, en los pesos de la función objetivo (`ConfiguracionPlanificacion`), en la capacidad nominal de los camiones (8 unidades) o en la dotación de carga invalida la política preentrenada y requiere ejecutar un nuevo ciclo de entrenamiento curricular (`python/scripts/training/train_rl_policy.py` / `python/scripts/rl/train_rl.py`).

---

<a id="metodologia-de-experimentacion-y-comparacion"></a>
## Metodología de Experimentación y Comparación

Para evaluar de manera imparcial los cinco algoritmos del sistema (`RL`, `HIBRIDO`, `GREEDY`, `RANDOM` y `GA`), el repositorio provee el ejecutor experimental `python/scripts/simulation/run_simulated_comparison.py`.

### Estructura Curricular y Dominio de Prueba
El entrenamiento y la evaluación abarcan cuatro regímenes de demanda operativa:
1. **Consolidación base:** 6 a 15 tareas por turno.
2. **Carga operativa típica:** 11 a 22 tareas por turno.
3. **Alta demanda y ventanas estrechas:** 14 a 30 tareas por turno.
4. **Dominio extendido:** 8 a 35 tareas por turno.

### Ejecución de Experimentos Comparativos

El script `run_simulated_comparison.py` permite evaluar lotes de turnos existentes o generar nuevos días sintéticos al vuelo, ejecutando múltiples réplicas por cada turno tanto en modo estimado como en el motor exportado de AnyLogic (vía Alpyne):

```powershell
Set-Location .\python

# Comparación rápida en modo estimado (sin requerir motor Java exportado)
..\.venv\Scripts\python.exe scripts/simulation/run_simulated_comparison.py --generar-dias 5 --repeticiones 3 --solo-estimado

# Comparación completa ejecutando el modelo exportado de AnyLogic vía Alpyne
..\.venv\Scripts\python.exe scripts/simulation/run_simulated_comparison.py --generar-dias 5 --repeticiones 3
```

### Trazabilidad de Resultados y Semillas

Cada corrida experimental crea un directorio aislado con formato `seed_<seed_inicial>_<YYYY-MM-DD_HH-MM-SS>` dentro de `python/data/results/`, almacenando:
- `metadata_experimento.json`: Parámetros completos del experimento, semillas por escenario y por algoritmo (`RL`, `HIBRIDO`, `GREEDY`, `RANDOM`, `GA`), e historial de turnos evaluados.
- `resumen_resultados.csv` y `resumen_resultados.json`: Tabla consolidada con los indicadores estimados y simulados de cada réplica y método:
  - Factibilidad y estado de terminación del motor.
  - Costo total estimado y costo total simulado.
  - Distancia total recorrida (`km`), duración del turno (`min`) y cantidad de viajes.
  - Cantidad de pedidos tardíos, minutos de tardanza acumulados y desbalance horario entre camiones.
  - Tiempo de cómputo del planificador (`ms`) y tiempo de ejecución del motor de simulación (`s`).

> [!NOTE]
> Los reportes comparativos deben conservar todas las réplicas y escenarios evaluados sin filtrado selectivo. En instancias con ventanas horarias altamente restrictivas o topologías puntuales, una heurística constructiva (`GREEDY`) o el refinamiento evolutivo (`HIBRIDO`) pueden alcanzar costos menores que la política `RL` pura; registrar dicho comportamiento forma parte del análisis experimental objetivo del gemelo digital.
