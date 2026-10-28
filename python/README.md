# Componente de Planificación en Python — Pedemonte Digital Twin

Este directorio contiene el motor de optimización y backend de planificación logística para **Pedemonte Digital Twin**, el cual se comunica de manera bidireccional con AnyLogic 8.9.9 PLE (Java) mediante PyPeline y Alpyne.

---

## 1. Requisitos Previos

- **Sistema Operativo:** Windows 10/11 (arquitectura nativa de AnyLogic y PyPeline en el proyecto).
- **Versión de Python:** **Python 3.11** (probado y validado en Python 3.11.9 de 64 bits).
- **Java Runtime:** Java 17 o superior (requerido para AnyLogic y ejecución de modelos exportados mediante Alpyne).
- **PowerShell:** Windows PowerShell 5.1 o PowerShell 7+.

---

## 2. Preparación del Entorno Virtual e Instalación

El repositorio utiliza un entorno virtual unificado en la raíz del proyecto (`.venv/`), accesible desde `python/` mediante la ruta relativa `..\.venv\Scripts\python.exe`.

### Paso 1: Clonar y posicionarse en el proyecto
```powershell
git clone <URL_DEL_REPOSITORIO> PedemonteDigitalTwin_v2
Set-Location PedemonteDigitalTwin_v2
```

### Paso 2: Crear el entorno virtual con Python 3.11
```powershell
py -3.11 -m venv .venv
# O si python apunta a la versión 3.11:
python -m venv .venv
```

### Paso 3: Activar el entorno e instalar dependencias
```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### Paso 4: Instalar todas las dependencias

El proyecto utiliza un único archivo `python/requirements.txt`
para ejecución, integración, entrenamiento RL, pruebas y análisis estático.

Desde la raíz del repositorio:

```powershell
.\.venv\Scripts\python.exe -m pip install -r python\requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

No es necesario instalar otros archivos requirements.

---

## 3. Estructura de la Carpeta `python/`

```text
python/
├── data/
│   ├── processed/
│   │   ├── demanda_geografica.csv    # Coordenadas y demanda real de Granadero Baigorria
│   │   ├── estadistica/              # Probabilidades de demanda, resumen municipal y gráficos
│   │   ├── extraccion/               # Salida estructurada de direcciones OCR y progreso
│   │   └── geocoding/                # Direcciones geolocalizadas (lat/lon) y descartadas
│   ├── raw/
│   │   └── hojas_pedidos/            # Imágenes manuscritas originales de pedidos escaneados
│   └── routing/
│       └── cache_vial.csv            # Matriz estricta de distancias y tiempos viales
├── models/
│   └── rl/
│       ├── rl_policies.json          # Manifiesto oficial y hashes de políticas RL
│       └── rl_policy.zip             # Pesos entrenados del modelo RL (pedemonte-rl-single-policy-v1)
├── planner/
│   ├── algorithms/                   # Algoritmos de resolución
│   │   ├── ga.py                     # Algoritmo Genético puro
│   │   ├── greedy.py                 # Algoritmo Goloso constructivo
│   │   ├── hybrid_rl_ga.py           # Algoritmo Híbrido (RL provee semilla, GA refina)
│   │   └── random_feasible.py        # Generador de asignaciones aleatorias factibles
│   ├── core/                         # Estructuras base, contratos y configuración
│   │   ├── base.py                   # Protocolo abstracto PlanificadorTurno
│   │   ├── config.py                 # Parámetros operativos (flotas, penalizaciones, tiempos)
│   │   └── schema.py                 # DTOs: InstanciaTurno, PedidoInput, PlanTurno, Viajes
│   ├── domain/                       # Lógica de dominio y reglas de negocio
│   │   ├── preprocess.py             # Split de pedidos (>8 un.), ventanas y homogeneidad
│   │   └── validator.py              # Validación estricta de factibilidad, capacidades y volcador
│   ├── integration/                  # Puentes de integración AnyLogic y formatos externos
│   │   ├── alpyne_codec.py           # Serialización vectorial de instancias y planes
│   │   ├── estimated_comparison.py   # Benchmark comparativo de los 5 algoritmos en memoria
│   │   ├── instance_vector_codec.py  # Codificación compacta base64/struct para PyPeline
│   │   ├── order_excel_import.py     # Parser robusto de planillas de pedidos
│   │   ├── order_import_bridge.py    # Puente PyPeline para importar pedidos a AnyLogic
│   │   ├── planner_selector.py       # Selector unificado de algoritmos con carga diferida de RL
│   │   ├── route_excel_archive.py    # Archivado de hojas de ruta generadas
│   │   ├── route_excel_bridge.py     # Puente PyPeline para exportar hojas de ruta desde AnyLogic
│   │   ├── route_excel_export.py     # Generador de planillas Excel de despacho
│   │   ├── selector_bridge.py        # Puente PyPeline principal de AnyLogic (Main)
│   │   ├── simulated_comparison.py   # Ejecución multi-instancia en simuladores AnyLogic
│   │   ├── simulated_comparison_bridge.py # Puente PyPeline para el experimento de comparación
│   │   └── simulated_execution.py    # Wrapper de control Alpyne para corridas de AnyLogic
│   ├── rl/                           # Entorno Gymnasium, observación y runtime de RL
│   │   ├── base_env.py               # Entorno Gymnasium base
│   │   ├── instance_generator.py     # Generador de instancias sintéticas y geo-distribuidas
│   │   ├── policy_config.py          # Configuración de máscaras y dimensiones de la red
│   │   ├── policy_env.py             # Entorno de política con máscaras de acción
│   │   ├── policy_estimator.py       # Estimador de costos operacionales para RL
│   │   ├── policy_planner.py         # Envoltorio de inferencia SB3 MaskablePPO
│   │   ├── policy_runtime.py         # Runtime operativo, verificación de hash y fallback
│   │   ├── reward.py                 # Función de recompensa alineada al costo estimado
│   │   ├── schedule_estimator.py     # Estimación rápida de cronogramas para recompensa
│   │   ├── training_cases.py         # Casos fijos de validación curricular
│   │   ├── training_curriculum.py    # Definición de fases curriculares de entrenamiento
│   │   ├── training_data.py          # Procesamiento y particionado de datos geográficos
│   │   ├── training_env.py           # Vectorización y callbacks de entrenamiento
│   │   ├── training_instance_generator.py # Generador curricular
│   │   └── training_validation.py    # Evaluación externa fija durante entrenamiento
│   └── routing/                      # Modelo vial, tiempos, operaciones y función objetivo
│       ├── decoder.py                # Decodificación de cronogramas y llegadas a clientes
│       ├── objective.py              # Función objetivo única determinista (costo operativo)
│       ├── operations.py             # Simulación de cargas, ventanas y tiempos de respuesta
│       ├── travel.py                 # Proveedores de distancias (Haversine y Caché Vial)
│       └── vial_cache.py             # Proveedor vial estricto basado en cache_vial.csv
├── scripts/
│   ├── data_processing/              # Pipeline de datos, OCR de pedidos y geocodificación
│   │   ├── estadistica_demanda.py    # Agregación y cálculo de probabilidades de demanda
│   │   ├── extraccion_ocr.py         # OCR multimodal con Gemini (Google AI Studio)
│   │   ├── geocoding.py              # Geocodificación OSM Nominatim de direcciones
│   │   ├── graficos_demanda.py       # Gráficos exploratorios (Top 15 barrios y donut municipal)
│   │   └── run_pipeline_demanda.py   # Orquestador del pipeline completo de demanda
│   ├── simulation/
│   │   ├── run_simulated_comparison.py     # CLI: corrida comparativa de los 5 métodos
│   │   └── run_single_simulated_execution.py # CLI: corrida de un plan en AnyLogic exportado
│   └── training/
│       └── train_rl_policy.py              # Script reproducible de reentrenamiento RL
├── templates/
│   ├── ejemplo_pedidos_importacion.xlsx    # Planilla de ejemplo con pedidos precargados
│   └── plantilla_pedidos_pedemonte.xlsx    # Plantilla en blanco para carga operativa
├── tests/                            # Suite automatizada (167 pruebas)
│   ├── algorithms/                   # Tests de Greedy, Random, GA e Híbrido
│   ├── data_processing/              # Tests de pipeline de demanda, OCR y geocodificación
│   ├── domain/                       # Tests de validación y preprocess (split de pedidos)
│   ├── integration/                  # Tests de serialización, PyPeline, Alpyne y Excel
│   ├── rl/                           # Tests de entornos, generadores y runtime RL
│   └── routing/                      # Tests de costo, caché vial y operaciones
├── pyrefly.toml                      # Configuración de tipado estático
└── requirements.txt                  # Dependencias de ejecución y desarrollo
```

---

## 4. Verificación y Ejecución de Pruebas

Toda modificación en el código debe validar el 100% de la suite de pruebas automatizadas:

```powershell
# Desde el directorio python/:
..\.venv\Scripts\python.exe -m pytest

# O desde la raíz del repositorio:
.venv\Scripts\python.exe -m pytest python/
```

**Resultado esperado:**
```text
============================= 167 passed in ~10s =============================
```

### Verificación de Tipado Estático (Pyrefly)
```powershell
Set-Location python
..\.venv\Scripts\python.exe -m pyrefly check planner
```
**Resultado esperado:** `INFO 0 errors`.

---

## 5. Comprobación del Modelo RL y Políticas

El sistema utiliza la política `pedemonte-rl-single-policy-v1` serializada en formato ZIP compatible con Stable-Baselines3 y PyTorch. Para garantizar que el modelo no ha sido modificado accidentalmente, se debe comprobar su hash criptográfico SHA-256:

```powershell
Get-FileHash python/models/rl/rl_policy.zip -Algorithm SHA256
```
**Hash oficial:**
```text
7D2838963F578656919822EE258E9C87C38257829EA801BBD3B41FCF26D20DA4
```

El archivo `python/models/rl/rl_policies.json` contiene la configuración de producción:
- Límite validado de tareas para la política RL: **12 tareas**.
- Espacio de acción: **30 posiciones** (máximas acciones seleccionables por paso, equivale al máximo de pedidos/tareas del turno).
- Límite general operativo del turno: **20 tareas**.

---

## 6. Configuración del Proveedor Vial Estricto

El archivo `python/data/routing/cache_vial.csv` contiene las distancias y tiempos de viaje reales obtenidos de la red vial de Granadero Baigorria y Rosario.

**Regla operativa:** Si se ingresan pedidos cuyas coordenadas no se encuentran registradas en la matriz, el proveedor vial detendrá la ejecución por seguridad para evitar estimaciones no validadas en producción. Para agregar nuevas direcciones, se deben incluir previamente los pares origen-destino en dicho archivo.

---

## 7. Integración con AnyLogic (PyPeline)

AnyLogic se comunica localmente mediante el conector **PyPeline** (agente `pyCommunicator`).

1. En AnyLogic, abrir el modelo `PedemonteDigitalTwin_v2.alp`.
2. En el agente `Main`, ubicar el componente `pyCommunicator`.
3. Configurar:
   - **`pythonCommandType`**: `PYTHON_PATH`
   - **`pythonExecPath`**: Ruta absoluta local a su ejecutable, por ejemplo:
     `C:\Users\<SU_USUARIO>\...\PedemonteDigitalTwin_v2\.venv\Scripts\python.exe`
4. Al iniciar la simulación, AnyLogic invoca automáticamente:
   - `selector_bridge.inicializar(...)` para preparar los algoritmos y la caché vial.
   - `selector_bridge.planificar_vector(...)` para resolver la logística del turno según el método elegido (GREEDY, RANDOM, GA, RL o HÍBRIDO).
   - `order_import_bridge.importar_pedidos_excel_pypeline_v2(...)` si se cargan pedidos desde Excel.
   - `route_excel_bridge.exportar_hoja_ruta_desde_anylogic(...)` para generar la hoja de ruta de los camiones.

> [!WARNING]
> La propiedad `pythonExecPath` de AnyLogic es personal de su máquina. No confirme (`commit`) cambios a esa propiedad en el archivo `.alp` para no invalidar las rutas de sus compañeros.

---

## 8. Scripts CLI de Simulación y Comparación

El componente incluye utilidades de línea de comandos para comparar algoritmos y validar escenarios sin abrir la interfaz gráfica de AnyLogic:

### Requisito para simulaciones exportadas

Los scripts de comparación simulada y ejecución individual utilizan
un modelo AnyLogic exportado, diferente de abrir y ejecutar el `.alp`
directamente desde AnyLogic.

La ruta predeterminada es:

`python/anylogic_export/simulated_comparison/PedemonteDigitalTwin_v2.zip`

Este archivo no se distribuye mediante Git porque el directorio
`anylogic_export/` está ignorado.

Cada desarrollador debe disponer de una exportación compatible
generada desde el modelo AnyLogic del proyecto.

Antes de ejecutar estos scripts desde `python/`, comprobar:

```powershell
Test-Path anylogic_export/simulated_comparison/PedemonteDigitalTwin_v2.zip
```

Si devuelve `False`, las comparaciones simuladas mediante Alpyne
no estarán disponibles hasta generar o incorporar esa exportación.

El flujo normal de planificación mediante PyPeline no depende de
estos scripts de simulación exportada.

**Limitación pendiente:** estos scripts conservan parámetros
históricos que deben sincronizarse en una revisión funcional
posterior con el límite operativo de 20 tareas y los pedidos mixtos.

Los siguientes ejemplos se ejecutan desde el directorio `python/`.

### Comparativa de los 5 algoritmos sobre un archivo Excel
```powershell
..\.venv\Scripts\python.exe scripts/simulation/run_simulated_comparison.py `
    --archivo-excel templates/ejemplo_pedidos_importacion.xlsx `
    --seed-escenario 6001 `
    --salida results/comparacion_ejemplo.json
```

### Ejecución de un único método en AnyLogic exportado
```powershell
..\.venv\Scripts\python.exe scripts/simulation/run_single_simulated_execution.py `
    --metodo HIBRIDO `
    --archivo-excel templates/ejemplo_pedidos_importacion.xlsx `
    --seed-escenario 6001
```

---

## 9. Archivos Protegidos y Buenas Prácticas de Git

### Archivos Protegidos (NO eliminar ni modificar sin autorización):
- `models/rl/rl_policy.zip` y `models/rl/rl_policies.json`.
- `data/routing/cache_vial.csv`.
- `data/processed/demanda_geografica.csv`.
- `data/processed/geocoding/direcciones_geolocalizadas.csv`.
- `data/processed/estadistica/probabilidades_demanda.csv`.
- `templates/ejemplo_pedidos_importacion.xlsx` y `templates/plantilla_pedidos_pedemonte.xlsx`.
- Modelos exportados en `anylogic_export/simulated_comparison/PedemonteDigitalTwin_v2.zip`.

### Archivos que NO deben comitearse (gestionados por `.gitignore`):
- Credenciales y variables locales: `.env`, `.env.*` (usar `.env.example`).
- Archivos de log generados: `*.log`, `alpynesimulated-*.log`, `modelsimulated-*.log`.
- Backups de edición: `*.bak`.
- Directorios temporales y de caché: `__pycache__/`, `.pytest_cache/`.
- Entornos virtuales locales: `.venv/`, `venv/`.
- Resultados de corridas locales: `results/*.json`, `results/*.csv`.

---

## 10. Pipeline de Demanda Histórica: OCR, Geocodificación y AnyLogic

Para alimentar la simulación estocástica en AnyLogic (generación de pedidos representativos a 100 días de operación), el proyecto cuenta con un pipeline completo de procesamiento de datos históricos ubicado en `scripts/data_processing/`.

### Arquitectura del Pipeline

```text
Imágenes Manuscritas (data/raw/hojas_pedidos/)
                      │
                      ▼ (extraccion_ocr.py - Google AI Studio Gemini API)
Direcciones Estructuradas (data/processed/extraccion/direcciones.json)
                      │
                      ▼ (geocoding.py - OpenStreetMap / Nominatim)
Coordenadas y Normalización (data/processed/geocoding/direcciones_geolocalizadas.csv)
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
Matriz de Probabilidades       Visualizaciones EDA
(estadistica_demanda.py)       (graficos_demanda.py)
        │                           │
        ▼                           ▼
probabilidades_demanda.csv     barrios_top15.png
resumen_ciudades.csv           pedidos_por_ciudad.png
        │
        ▼
Generación Estocástica de Pedidos para 100 días en AnyLogic
```

### Configuración de la API Key

1. Copiá la plantilla de variables de entorno:
   ```powershell
   Copy-Item .env.example .env
   ```
2. Editá `.env` y asigná tu API Key de Google AI Studio:
   ```text
   GEMINI_API_KEY=tu_api_key_aqui
   ```
   *(El archivo `.env` está protegido por `.gitignore` y nunca se sube al repositorio).*

### Ejecución de los Scripts

Podés ejecutar cada etapa individualmente o todo el pipeline mediante `run_pipeline_demanda.py`:

```powershell
# Ejecutar todo el pipeline:
..\.venv\Scripts\python.exe scripts/data_processing/run_pipeline_demanda.py --etapa all

# O ejecutar únicamente el cálculo estadístico y gráficos:
..\.venv\Scripts\python.exe scripts/data_processing/run_pipeline_demanda.py --etapa estadistica
..\.venv\Scripts\python.exe scripts/data_processing/run_pipeline_demanda.py --etapa graficos
```

