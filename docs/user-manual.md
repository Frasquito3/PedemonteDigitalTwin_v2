<div align="center">

# 🎛️ Manual de Uso y Operación

[🏠 Inicio](../README.md) •
[📚 Documentación](README.md) •
[🚀 Comenzar](getting-started.md) •
[🏗️ Sistema](system-design.md) •
**🎛️ Operación** •
[🧪 Calidad](quality-and-experiments.md) •
[🛠️ Desarrollo](development-and-deployment.md)

</div>

---

## 1. Configuración Inicial del Escenario

Para arrancar la operativa, el primer paso es definir las condiciones globales de la ejecución:
- **Turno:** `MAÑANA` (07:30 a 12:00) o `TARDE` (14:00 a 17:00). Recordá que el sistema ofrece **15 minutos de tolerancia operativa** al finalizar (ej. los camiones pueden regresar al corralón hasta las 12:15 sin recibir una penalidad extrema).
- **Semilla (Opcional):** Para reproducir escenarios idénticos (por ejemplo, para el método Random o GA).

---

## 2. Carga de Pedidos

El sistema admite pedidos manuales y carga masiva mediante archivos `.xlsx`. 
*La capacidad técnica productiva demostrada para el RL es de hasta 12 pedidos.* Si un pedido supera las 8 unidades, el sistema aplicará un *Split Automático* en múltiples tareas con el mismo ID y las mismas reglas de volcador.

### 📄 Carga por Excel (.xlsx)
El sistema tomará automáticamente la hoja llamada `Pedidos`. Si no existe, tomará la hoja activa. Los encabezados no son *case-sensitive*, pero deben coincidir con la convención:

| Dato Requerido | Formatos de Encabezado Aceptados | Formato de Celda |
| :--- | :--- | :--- |
| **ID del Pedido** | `ID`, `Código`, `ID pedido` | Texto o Número |
| **Cantidad** | `Unidades`, `Cantidad`, `Carga` | Número Entero (>0) |
| **Latitud** | `Latitud`, `Lat` | Número Decimal |
| **Longitud** | `Longitud`, `Lon`, `Lng` | Número Decimal |
| **Usa Volcador?** | `Requiere volcador`, `Volcador` | Booleano: `Sí/No`, `True/False`, `1/0` |
| **Usa Ventana?** | `Tiene ventana`, `Ventana` | Booleano |
| **Hora Desde** | `Hora desde`, `Desde` | Texto (`HH:mm`) u Hora Excel |
| **Hora Hasta** | `Hora hasta`, `Hasta` | Texto (`HH:mm`) u Hora Excel |

---

## 3. Generación del Plan

1. Una vez cargados los pedidos, presioná **Preparar Operación**. Esto consolida las cargas, realiza el split y verifica las coordenadas contra la caché vial.
2. Seleccioná un método en la lista desplegable:
   - `RL` (Recomendado)
   - `HÍBRIDO`
   - `GREEDY` / `RANDOM` / `GA`
3. Presioná **Generar y Ejecutar Plan**. El sistema mostrará en la consola el *Costo Estimado* y comenzará a mover los camiones en el GIS.

---

## 4. Visualización en el Dashboard (GIS)

Podés controlar la vista satelital desde el menú de la izquierda:
- 🗺️ **General:** Vista amplia del corralón y de todas las rutas de los clientes.
- 🚛 **Camión 0:** La cámara sigue exclusivamente los movimientos de este camión en tiempo real.
- 🚛 **Camión 1:** La cámara sigue exclusivamente a este camión.

Los estados transicionales de los camiones irán rotando desde `En corralón`, `Cargando`, `Viajando a <pedido>`, `En cliente <pedido>` hasta `Regresando al corralón`. 

Una vez que ambos finalicen, el estado mostrará **Plan finalizado** y revelará las métricas finales (Costo real, tardanzas y kilómetros). 

> [!TIP]
> Antes de cargar un nuevo Excel, **debés presionar el botón "Limpiar Operación"**. No intentes reutilizar una simulación a la mitad.

---

## 🛑 Solución de Problemas (Troubleshooting)

<details>
<summary><strong>❌ Pypeline no inicia o no encuentra Python</strong></summary>

En el agente `Main` (objeto `pyCommunicator`), verificá la propiedad `pythonExecPath`. Debe apuntar a un path local válido (ej. `C:\Ruta\A\Tu\Repositorio\.venv\Scripts\python.exe`). Si actualizaste repositorios, asegurate de no haber *pusheado* tu ruta local absoluta.
</details>

<details>
<summary><strong>❌ Error de "Cache Miss" al iniciar la operativa</strong></summary>

El sistema utiliza una caché vial estricta. Si ingresás unas coordenadas de un cliente que no existen en la matriz (`cache_vial.csv`), Python detendrá la ejecución por seguridad. Deberás añadir el tramo al CSV manualmente o usar las coordenadas probadas previamente. No actives variables locales de *fallback* vial en productivo.
</details>

<details>
<summary><strong>❌ La planilla se rechaza y no carga nada</strong></summary>

1. Confirmá que los IDs no estén duplicados antes del split.
2. Comprobá las ventanas horarias: "Hora inicial" no puede ser mayor a la "Hora final", y ambas deben estar dentro de las franjas de turno (07:30 a 12:00 o 14:00 a 17:00).
3. Asegurate de que los números no tengan formatos extraños.
</details>
