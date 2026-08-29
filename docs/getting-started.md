<div align="center">

# 🚀 Primeros Pasos

[🏠 Inicio](../README.md) •
[📚 Documentación](README.md) •
**🚀 Comenzar** •
[🏗️ Sistema](system-design.md) •
[🎛️ Operación](user-manual.md) •
[🧪 Calidad](quality-and-experiments.md) •
[🛠️ Desarrollo](development-and-deployment.md)

</div>

---

Esta guía te ayudará a configurar el entorno local, validarlo y ejecutar la primera simulación del Gemelo Digital.

## 📋 Requisitos Previos

- **Sistema Operativo:** Windows
- **Terminal:** PowerShell
- **Lenguaje:** Python 3.11
- **Simulador:** AnyLogic (Personal Learning Edition o superior)
- **Control de versiones:** Git

## 🛠️ Instalación y Configuración

### 1. Clonar el repositorio
Abrí PowerShell y cloná el repositorio:
```powershell
git clone <URL-DEL-REPOSITORIO>
Set-Location .\PedemonteDigitalTwin_v2
```

### 2. Crear y activar el Entorno Virtual (VENV)
Es **fundamental** aislar las dependencias del proyecto. Estando en la raíz del repositorio:
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r .\python\requirements-dev.txt
```

> [!TIP]
> El archivo `requirements-dev.txt` incluye internamente a `requirements-rl.txt`, por lo que instalará todas las dependencias necesarias: `numpy`, `gymnasium`, `stable-baselines3`, `pyrefly`, `anylogic-alpyne`, `openpyxl`, y `pytest`.

### 3. Validar el Checkpoint RL
El repositorio ya incluye un checkpoint entrenado y productivo. Para validar que está intacto, comprobá su hash:

```powershell
Get-FileHash ".\python\models\rl\rl_policy.zip" -Algorithm SHA256
```
El hash esperado debe ser exactamente: `7D2838963F578656919822EE258E9C87C38257829EA801BBD3B41FCF26D20DA4`.

### 4. Ejecutar la Regresión de Pruebas
Validá que toda la lógica de planificación de Python esté intacta:

```powershell
Set-Location .\python
$py = (Resolve-Path "..\.venv\Scripts\python.exe").Path
& $py -m pytest -q --tb=short
```
> Resultado esperado: `76 passed`.

---

## 🖥️ Conexión con AnyLogic

El puente entre la interfaz de simulación y el planificador local se hace mediante **Pypeline**.

1. Abrí AnyLogic y cargá el modelo: `anylogic/PedemonteDigitalTwin_v2/PedemonteDigitalTwin_v2.alp`.
2. En la vista del agente `Main`, buscá el objeto `pyCommunicator`.
3. Revisá las propiedades del objeto y configurá:
   - **pythonCommandType**: `PYTHON_PATH`
   - **pythonExecPath**: Debés poner la ruta **absoluta** a tu ejecutable de Python. Ejemplo: `C:\Users\tu-usuario\PedemonteDigitalTwin_v2\.venv\Scripts\python.exe`.
   
> [!WARNING]
> La ruta a Python es personal de tu equipo. Cuando vayas a guardar cambios en AnyLogic y hacer *commit*, tené mucho cuidado de no subir este cambio de ruta, o romperás el entorno de tus compañeros.

Una vez configurado, presioná **Build Model** en AnyLogic y asegurate de que no haya errores de compilación.

---

## 🎮 Primera Ejecución

1. Iniciá la simulación desde AnyLogic.
2. En el Dashboard principal, cargá pedidos de forma manual o importa un `.xlsx`.
3. Presioná el botón <kbd>Preparar operación</kbd>.
4. En el selector de métodos, elegí `RL`.
5. Presioná <kbd>Generar y ejecutar plan</kbd>.
6. Usá los botones <kbd>General</kbd>, <kbd>Camión 0</kbd> y <kbd>Camión 1</kbd> para hacer el seguimiento GIS interactivo de la ejecución operativa.

🎉 **¡Listo!** Ya tenés el entorno funcionando. Para aprender más sobre la interacción y la carga masiva de datos, revisá el [Manual de uso](user-manual.md).
