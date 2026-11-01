<div align="center">

# Guía de Instalación y Primeros Pasos

[Inicio](../README.md) •
[Documentación](README.md) •
**Primeros Pasos** •
[Diseño del Sistema](system-design.md) •
[Manual de Operación](user-manual.md) •
[Calidad y Experimentos](quality-and-experiments.md) •
[Desarrollo y Despliegue](development-and-deployment.md)

</div>

---

Esta guía describe el procedimiento para configurar el entorno de desarrollo local, verificar la integridad de los artefactos del proyecto y ejecutar la simulación del gemelo digital.

## Requisitos Previos

- **Sistema Operativo:** Windows 10/11 (para ejecución interactiva con AnyLogic Desktop) o Linux/macOS (para ejecución exclusiva del núcleo Python y pruebas).
- **Terminal:** PowerShell o Bash.
- **Intérprete Python:** Python 3.11 o superior.
- **Simulador:** AnyLogic 8 Desktop (Personal Learning Edition, University o Professional).
- **Control de versiones:** Git.

---

<a id="instalacion-y-configuracion"></a>
## Instalación y Configuración

### 1. Clonar el repositorio

Desde una terminal en el directorio de trabajo deseado:

```powershell
git clone https://github.com/Frasquito3/PedemonteDigitalTwin_v2.git
Set-Location .\PedemonteDigitalTwin_v2
```

### 2. Crear y activar el entorno virtual

Se recomienda aislar todas las dependencias del proyecto dentro de un entorno virtual `.venv` en la raíz del repositorio:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r .\python\requirements.txt
```

> [!NOTE]
> El archivo `python/requirements.txt` instala las dependencias de planificación, aprendizaje por refuerzo, procesamiento de datos y calidad de código: `numpy`, `gymnasium`, `stable-baselines3`, `sb3-contrib`, `anylogic-alpyne`, `openpyxl`, `pandas`, `matplotlib`, `geopy`, `google-genai`, `python-dotenv`, `pytest`, `pyrefly`, `ruff` y `pdoc`.
>
> En caso de utilizar la etapa de extracción OCR de remitos manuscritos mediante Google GenAI, se debe crear el archivo `.env` a partir de la plantilla provista:
> ```powershell
> Copy-Item .env.example .env
> # Definir la variable GEMINI_API_KEY con una clave válida de Google AI Studio
> ```

### 3. Activar los hooks locales de Git (Opcional)

Para validar automáticamente el formato y las reglas de estilo con `ruff` antes de cada `commit` y `push`:

```powershell
git config core.hooksPath .githooks
```

### 4. Verificar la integridad del checkpoint RL

El repositorio incluye la política preentrenada productiva y su manifiesto en `python/models/rl/rl_policies.json`. Para comprobar localmente la suma de verificación SHA-256 del archivo `rl_policy.zip`:

```powershell
Get-FileHash ".\python\models\rl\rl_policy.zip" -Algorithm SHA256
```

El hash SHA-256 esperado debe coincidir con el declarado en `rl_policies.json`:

```text
DF1EDC421763EAA7AA426C596AE16998D5867DDC720EDA8D2CF88923F5FE434F
```

### 5. Ejecutar las pruebas automatizadas y el chequeo estático

Para comprobar que el núcleo de planificación, los códecs de integración y el pipeline de datos funcionan correctamente:

```powershell
Set-Location .\python
..\.venv\Scripts\python.exe -m pytest -q --tb=short
..\.venv\Scripts\pyrefly.exe check --python-interpreter-path ..\.venv\Scripts\python.exe
..\.venv\Scripts\ruff.exe check .
```

Resultado esperado: `174 passed` en `pytest` y `0 errors` tanto en `pyrefly` como en `ruff`.

---

## Vinculación con AnyLogic

La comunicación entre el modelo de simulación en AnyLogic y el planificador en Python se realiza mediante **Pypeline** y la interfaz web integrada con **AnyLogicWebToolkit**.

### 1. Importar librerías externas en AnyLogic

1. Descargar los archivos `.jar` desde sus repositorios oficiales:
   - [AnyLogic-Pypeline](https://github.com/the-anylogic-company/AnyLogic-Pypeline/releases)
   - [AnyLogicWebToolkit](https://github.com/NiconiKimg/AnyLogicWebToolkit)
2. En AnyLogic, abrir la vista **Palette**, pulsar **Manage Libraries** $\rightarrow$ **Add** y seleccionar ambos archivos `.jar`.

### 2. Configurar el conector Python (`pyCommunicator`)

1. Abrir el modelo ubicado en `anylogic/PedemonteDigitalTwin_v2/PedemonteDigitalTwin_v2.alp`.
2. En el agente `Main`, seleccionar el objeto `pyCommunicator`.
3. Verificar y ajustar sus propiedades:
   - **pythonCommandType:** `PYTHON_PATH`
   - **pythonExecPath:** Ruta absoluta al ejecutable Python del entorno virtual local, por ejemplo:
     ```text
     C:\Users\usuario\PedemonteDigitalTwin_v2\.venv\Scripts\python.exe
     ```

> [!WARNING]
> La propiedad `pythonExecPath` contiene una ruta local específica de cada estación de trabajo. Al guardar cambios sobre el archivo `.alp`, se debe verificar mediante `git diff` no incluir rutas locales antes de realizar un `commit`.

### 3. Exportación para simulación headless con Alpyne (Opcional)

Si se desea ejecutar el banco de pruebas comparativo automatizado desde Python (`python/scripts/simulation/run_simulated_comparison.py`), se debe exportar el experimento desde AnyLogic hacia la ruta:

```text
python/anylogic_export/simulated_comparison/PedemonteDigitalTwin_v2.zip
```

---

## Primera Ejecución Interactiva

1. Compilar el modelo en AnyLogic (**Build Model**) y verificar que no existan errores de compilación.
2. Iniciar la simulación (**Run**).
3. En el panel operativo, ingresar pedidos manualmente o importar una planilla `.xlsx` (generada desde la interfaz web de pedidos o desde el generador estocástico).
4. Presionar **Preparar operación** para validar los pedidos, ejecutar la partición por capacidad (*order split*) y verificar la caché vial.
5. Seleccionar el algoritmo deseado (`RL`, `HIBRIDO`, `GREEDY`, `RANDOM` o `GA`) y presionar **Generar y ejecutar plan**.
6. Utilizar los controles de cámara (**General**, **Camión 0** y **Camión 1**) para inspeccionar el recorrido en el mapa GIS y consultar los indicadores de desempeño al concluir el turno.

Para mayores detalles sobre los formatos de entrada y el uso de la interfaz, consultar el [Manual de Operación](user-manual.md).
