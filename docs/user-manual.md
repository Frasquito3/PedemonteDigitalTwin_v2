<div align="center">

# Manual de Uso y Operación

[Inicio](../README.md) •
[Documentación](README.md) •
[Primeros Pasos](getting-started.md) •
[Diseño del Sistema](system-design.md) •
**Manual de Operación** •
[Calidad y Experimentos](quality-and-experiments.md) •
[Desarrollo y Despliegue](development-and-deployment.md)

</div>

---

## 1. Configuración Inicial del Escenario

Antes de generar un plan de despacho, se deben definir los parámetros operativos del turno:

- **Turno operativo:**
  - `MANANA`: de 07:30 a 12:00 (con tolerancia operativa hasta las 12:15).
  - `TARDE`: de 14:00 a 17:00 (con tolerancia operativa hasta las 17:15).
  - Toda finalización posterior al límite de tolerancia de 15 minutos incurre en una penalización severa en la función objetivo.
- **Semilla de escenario y ejecución (Opcional):** Permite fijar la secuencia pseudoaleatoria tanto para los métodos estocásticos (`RANDOM`, `GA`, `HIBRIDO`) como para las variables estocásticas de simulación en AnyLogic.

---

<a id="2-carga-y-generacion-de-pedidos"></a>
## 2. Carga y Generación de Pedidos

El sistema admite el ingreso manual de pedidos desde el formulario de simulación, la generación de planillas desde la aplicación web local y la importación masiva mediante archivos Excel (`.xlsx`).

- **Límite operativo de tareas:** La política productiva de Aprendizaje por Refuerzo (`rl_policy.zip`) se encuentra entrenada y validada para instancias de **hasta 35 tareas operativas por turno** (`max_pedidos_rl_validado = 35`).
- **Partición automática (*Order Split*):** La capacidad máxima de cada camión es de **8 unidades**. Si un pedido ingresado supera las 8 unidades, el preprocesador lo divide automáticamente en múltiples tareas (`numero_parte` / `total_partes`) conservando la referencia al `pedido_original_id`, las coordenadas, la ventana horaria y la restricción de descarga por volcador.

### 2.1. Interfaz Web de Gestión de Pedidos

El directorio `anylogic/PedemonteDigitalTwin_v2/web/pedidos/` contiene una aplicación web (`pedidos.html`) integrable con AnyLogic y ejecutable en navegador que permite:
1. Buscar y geocodificar direcciones dentro del área de cobertura o seleccionar ubicaciones directamente sobre el mapa interactivo Leaflet.
2. Definir unidades de carga, requerimiento de volcador, turno y ventana horaria específica, o cargar un lote de demostración mediante el botón **Demo**.
3. Exportar la nómina validada a un archivo Excel con marca temporal en el nombre (`pedidos_YYYY-MM-DD_HH-mm-ss.xlsx`), evitando sobrescribir exportaciones previas.

### 2.2. Especificación del Formato Excel (`.xlsx`)

Al importar un archivo `.xlsx`, el importador (`planner.integration.order_excel_import`) selecciona la hoja denominada `Pedidos` o, en su defecto, la primera hoja activa con datos. Los nombres de columna se normalizan sin distinción de mayúsculas, minúsculas ni tildes:

| Campo | Encabezados Aceptados | Tipo y Restricciones |
| :--- | :--- | :--- |
| **Identificador** | `pedido_id`, `id`, `codigo`, `id_pedido` | Texto o entero único por fila. |
| **Cliente** | `cliente`, `nombre_cliente` | Texto (opcional). |
| **Dirección** | `direccion`, `domicilio`, `calle` | Texto descriptivo (opcional). |
| **Barrio / Localidad** | `barrio`, `zona`, `localidad` | Texto (opcional). |
| **Unidades** | `unidades`, `cantidad`, `carga`, `unidades_capacidad` | Entero positivo ($\ge 1$). |
| **Latitud** | `latitud`, `lat` | Decimal en rango $[-90, 90]$. |
| **Longitud** | `longitud`, `lon`, `lng`, `long` | Decimal en rango $[-180, 180]$. |
| **Requiere Volcador** | `requiere_volcador`, `volcador`, `usa_volcador` | Booleano (`SI`/`NO`, `True`/`False`, `1`/`0`). |
| **Tiene Ventana** | `tiene_ventana`, `ventana`, `ventana_horaria` | Booleano (`SI`/`NO`, `True`/`False`, `1`/`0`). |
| **Hora Desde** | `hora_desde`, `desde`, `inicio_ventana` | Formato `HH:MM` o tipo hora nativo de Excel. |
| **Hora Hasta** | `hora_hasta`, `hasta`, `fin_ventana` | Formato `HH:MM` o tipo hora nativo de Excel. |
| **Observaciones** | `observaciones`, `notas`, `comentario` | Texto libre (opcional). |

---

## 3. Generación y Ejecución del Plan

1. Una vez cargados los pedidos, presionar **Preparar operación**. El sistema ejecuta el preprocesamiento, particiona los pedidos mayores a 8 unidades y valida que todos los pares de coordenadas existan en la caché vial (`cache_vial.csv`).
2. Seleccionar el algoritmo de planificación:
   - `RL`: Inferencia mediante política neuronal con máscara temporal dura y selección multi-rollout.
   - `HIBRIDO`: Sembrado inicial con la solución de `RL` seguido de refinamiento evolutivo con `GA`.
   - `GA`: Algoritmo Genético por permutaciones con cruce Order Crossover (OX).
   - `GREEDY`: Heurística constructiva voraz guiada por urgencia horaria y cercanía.
   - `RANDOM`: Generador estocástico factible utilizado como línea base de contraste.
3. Presionar **Generar y ejecutar plan** para visualizar el costo estimado del plan y dar inicio a la simulación de despacho en el mapa GIS.
4. Asimismo, desde la pestaña de **Comparación**, es posible ejecutar los cinco algoritmos sobre la misma instancia para contrastar sus métricas estimadas y simuladas.

---

<a id="4-visualizacion-en-el-panel-y-gis"></a>
## 4. Visualización en el Panel y GIS

Durante la corrida de simulación, el panel lateral permite alternar el encuadre de visualización geográfica:
- **General:** Vista integral del corralón y de la totalidad de los puntos de entrega del turno.
- **Camión 0:** Seguimiento centrado en tiempo real sobre el primer vehículo de la flota.
- **Camión 1:** Seguimiento centrado en tiempo real sobre el segundo vehículo de la flota.

El estado operativo de cada unidad progresa a través de las fases: `En corralón`, `Cargando`, `Viajando a <pedido>`, `En cliente <pedido>` y `Regresando al corralón`. Al concluir todos los viajes programados, el sistema informa **Plan finalizado** y consolida los indicadores definitivos de costo total, kilómetros recorridos, cantidad de viajes, tardanzas y desbalance de cierre entre camiones.

> [!IMPORTANT]
> Antes de importar un nuevo lote de pedidos o cambiar los parámetros de entrada tras una ejecución, se debe presionar **Limpiar operación** para restablecer el estado de la flota y del entorno.

---

<a id="5-resolucion-de-problemas-troubleshooting"></a>
## 5. Resolución de Problemas (Troubleshooting)

<details>
<summary><strong>Pypeline no inicia o no localiza el intérprete de Python</strong></summary>

En el agente `Main` de AnyLogic, inspeccionar el objeto `pyCommunicator` y verificar que la propiedad `pythonExecPath` apunte a la ruta absoluta del ejecutable `.venv\Scripts\python.exe` local y que `pythonCommandType` esté configurado en `PYTHON_PATH`.
</details>

<details>
<summary><strong>Error de "Cache Miss" al preparar la operación</strong></summary>

El planificador opera en modo de caché vial estricta (`permitir_fallback_vial=False`) para garantizar reproducibilidad. Si se ingresan coordenadas que no registran tramos precalculados en `python/data/routing/cache_vial.csv`, la validación detiene el proceso e informa el par faltante. Se deben emplear coordenadas dentro del catálogo geográfico validado o actualizar previamente la matriz vial.
</details>

<details>
<summary><strong>Rechazo de planilla Excel durante la importación</strong></summary>

1. Verificar que no existan identificadores (`pedido_id`) duplicados en la planilla de origen.
2. Comprobar la consistencia de las ventanas horarias: `hora_desde` debe ser estrictamente menor o igual a `hora_hasta`, y ambas deben encontrarse dentro del rango horario del turno seleccionado (`07:30`–`12:00` para mañana o `14:00`–`17:00` para tarde).
3. Confirmar que el campo `unidades` contenga números enteros positivos y que el total de tareas resultantes tras el *split* no supere el límite del selector.
</details>
