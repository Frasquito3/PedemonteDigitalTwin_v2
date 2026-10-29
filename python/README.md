# Pedemonte Digital Twin - Módulo Python

Este subproyecto contiene el núcleo algorítmico, el procesamiento de demanda histórica, los modelos de ruteo estocástico, el selector dinámico de políticas y los puentes de integración con la simulación en AnyLogic.

---

## Estructura del Módulo

```text
python/
├── data/
│   ├── processed/
│   │   ├── demanda_geografica.csv    # Dataset consolidado de pedidos históricos limpios
│   │   ├── estadistica/              # Probabilidades de demanda y resúmenes por zona
│   │   ├── extraccion/               # Salidas intermedias JSON de pedidos manuscritos
│   │   ├── geocoding/                # Direcciones geocodificadas y descartadas
│   │   └── pedidos_generados/        # Días de pedidos sintéticos generados en CSV y Excel
│   ├── raw/
│   │   └── hojas_pedidos/            # Hojas de pedidos escaneadas de corralón
│   └── routing/
│       └── cache_vial.csv            # Matriz fija de distancias y tiempos de AnyLogic
├── models/
│   └── rl/
│       ├── rl_policies.json          # Metadatos del modelo RL empaquetado
│       └── rl_policy.zip             # Modelo MaskablePPO entrenado
├── planner/
│   ├── algorithms/                   # Implementaciones de optimización
│   │   ├── ga.py                     # Algoritmo Genético con cruce OX y mutación swap/inversión
│   │   ├── greedy.py                 # Algoritmo Goloso determinista de referencia
│   │   ├── hybrid_rl_ga.py           # Algoritmo Híbrido: RL si tamaño ≤ 12, GA en caso contrario
│   │   └── random_feasible.py        # Generador aleatorio uniforme de planes factibles
│   ├── core/                         # Tipos fundamentales y constantes
│   │   ├── base.py                   # Protocolo BasePlanner y clases de resultado
│   │   ├── config.py                 # Parámetros operativos de corralón
│   │   └── schema.py                 # Dataclasses de pedidos, turnos y planes
│   ├── domain/                       # Reglas operativas del corralón
│   │   ├── preprocess.py             # Split automático de pedidos > 8 unidades y sanitización
│   │   └── validator.py              # Validador de factibilidad operativa y ventanas
│   ├── integration/                  # Puentes con AnyLogic (Pypeline / Alpyne) y Excel
│   │   ├── alpyne_codec.py           # Serialización de vectores fijos para Alpyne
│   │   ├── estimated_comparison.py   # Benchmark offline estimado entre algoritmos
│   │   ├── instance_vector_codec.py  # Empaquetado de instancias a arreglos primitivos
│   │   ├── order_excel_import.py     # Importador y validador de hojas de pedidos Excel
│   │   ├── planner_selector.py       # Selector dinámico de métodos de ruteo
│   │   ├── route_excel_archive.py    # Almacenamiento histórico de planes
│   │   ├── route_excel_export.py     # Exportador de hojas de ruta detalladas a Excel
│   │   ├── selector_bridge.py        # Puente para llamadas desde Java/Pypeline
│   │   ├── simulated_comparison.py   # Comparación ejecutada en AnyLogic exportado
│   │   └── simulated_execution.py    # Lanzador de AnyLogic standalone vía headless Java
│   ├── rl/                           # Reinforcement Learning (Gymnasium + Stable-Baselines3)
│   │   ├── base_env.py               # Entorno base Gymnasium para secuenciamiento
│   │   ├── instance_generator.py     # Generador de instancias sintéticas y reales
│   │   ├── policy_env.py             # Entorno con recompensas y métricas de validación
│   │   ├── policy_planner.py         # Planificador que infiere acciones con MaskablePPO
│   │   ├── reward.py                 # Cálculo de recompensas densas y terminales
│   │   ├── schedule_estimator.py     # Estimación rápida de cronogramas para el reward
│   │   ├── training_cases.py         # Casos fijos de prueba para seguimiento de entrenamiento
│   │   ├── training_curriculum.py    # Currículum de entrenamiento en etapas progresivas
│   │   ├── training_data.py          # Particionamiento Train/Validation/Test de demanda real
│   │   ├── training_env.py           # Wrapper de Gymnasium compatible con SB3
│   │   ├── training_instance_generator.py # Generador con probabilidades por etapa de currículum
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
│   │   ├── generar_pedidos_dia.py    # Generador estocástico de N días de pedidos a partir de históricos CSV
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
├── tests/                            # Suite automatizada (171 pruebas)
│   ├── algorithms/                   # Tests de Greedy, Random, GA e Híbrido
│   ├── data_processing/              # Tests de pipeline de demanda, OCR, geocodificación y generador
│   ├── domain/                       # Tests de validación y preprocess (split de pedidos)
│   ├── integration/                  # Tests de puentes Pypeline, Alpyne, Excel y Selector
│   ├── rl/                           # Tests de entornos Gym, recompensas y generadores RL
│   └── routing/                      # Tests de matriz vial, tiempos y función objetivo
├── pyrefly.toml                      # Configuración de formateo y estilo
└── requirements.txt                  # Dependencias congeladas del entorno
```

---

## Flujo de Datos Históricos y Pipeline de Demanda

El pipeline de demanda procesa las hojas manuscritas originales hasta convertirlas en una distribución empírica de pedidos georreferenciados:

```text
python/data/raw/hojas_pedidos/ (*.jpeg)
        │
        ▼ (extraccion_ocr.py - Google AI Studio Gemini API)
python/data/processed/extraccion/direcciones.json
        │
        ▼ (geocoding.py - OpenStreetMap Nominatim)
python/data/processed/geocoding/direcciones_geolocalizadas.csv
        │
        ├───────────────────────────┐
        ▼                           ▼
probabilidades_demanda.csv     barrios_top15.png
resumen_ciudades.csv           pedidos_por_ciudad.png
        │
        ▼
Generación Estocástica de Nuevos Pedidos (generar_pedidos_dia.py)
        │
        ▼ (.csv / .xlsx)
AnyLogic / Digital Twin / Optimizador
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

### Ejecución de los Scripts de Demanda

Podés ejecutar cada etapa individualmente o todo el pipeline mediante `run_pipeline_demanda.py`:

```powershell
# Ejecutar todo el pipeline completo (requiere API key para OCR):
python scripts/data_processing/run_pipeline_demanda.py --etapa all

# Ejecutar únicamente el cálculo estadístico y gráficos:
python scripts/data_processing/run_pipeline_demanda.py --etapa estadistica
python scripts/data_processing/run_pipeline_demanda.py --etapa graficos

# Generar N días de nuevos pedidos a partir del histórico:
python scripts/data_processing/run_pipeline_demanda.py --etapa generar -n 5 -p 8-12
```

### Generación de Nuevos Pedidos para Nuevos Días (`generar_pedidos_dia.py`)

El script `python/scripts/data_processing/generar_pedidos_dia.py` permite crear $N$ días de pedidos sintéticos calibrados con la operación real de Corralón Pedemonte, muestreando las direcciones históricas según su frecuencia empírica:

```powershell
# Generar 5 días con entre 8 y 12 pedidos por día (en formatos CSV y Excel):
python scripts/data_processing/generar_pedidos_dia.py -n 5 -p 8-12

# Generar 1 día con exactamente 10 pedidos para el turno TARDE:
python scripts/data_processing/generar_pedidos_dia.py -n 1 -p 10 --turno TARDE

# Generar 10 días, especificando semilla y guardando un archivo consolidado:
python scripts/data_processing/generar_pedidos_dia.py -n 10 -p 10 --seed 42 --consolidado

# Opciones disponibles:
#   -n, --dias           Cantidad N de días/escenarios a generar (por defecto: 1)
#   -p, --pedidos-por-dia Cantidad fija (ej. 10) o rango (ej. 8-12) de pedidos por día
#   --input-file         Ruta al CSV histórico (por defecto: demanda_geografica.csv)
#   -o, --output-dir     Carpeta de salida (por defecto: data/processed/pedidos_generados)
#   --formato            Formato de salida: csv, xlsx o ambos (por defecto: ambos)
#   --turno              MANANA, TARDE o MIXTO (por defecto: MANANA)
#   --consolidado        Crea además un archivo con todos los días unificados
#   --seed               Semilla para reproducibilidad estricta
```
