<div align="center">

# 📚 Centro de Documentación

<p>
  <img alt="Guías" src="https://img.shields.io/badge/Guías-5_Documentos-1565C0?style=for-the-badge">
</p>

[🏠 Inicio](../README.md) •
**📚 Documentación** •
[🚀 Comenzar](getting-started.md) •
[🏗️ Sistema](system-design.md) •
[🎛️ Operación](user-manual.md) •
[🧪 Calidad](quality-and-experiments.md) •
[🛠️ Desarrollo](development-and-deployment.md)

</div>

---

La documentación de Pedemonte Digital Twin está diseñada para responder a diferentes perfiles técnicos y operativos. El código, las pruebas y los manifiestos (hashes) son la única **fuente de verdad ejecutable**. 

Cualquier cambio de lógica en el repositorio debe ir acompañado de una actualización en el documento correspondiente.

## 🗺️ Mapa Documental

| Documento | Descripción Principal |
| :--- | :--- |
| **[🚀 Primeros pasos](getting-started.md)** | Instalación de dependencias (Python 3.11, `.venv`), configuración de Pypeline en AnyLogic y primera ejecución básica. |
| **[🏗️ Diseño del sistema](system-design.md)** | Detalles profundos de arquitectura. Incluye el rol de RL como propuesta central, el funcionamiento del refinamiento Híbrido, la caché vial y la función objetivo compartida. |
| **[🎛️ Manual de uso](user-manual.md)** | Guía de operación del usuario: configuración de turnos, importación de Excel, variables aceptadas, uso del Dashboard, navegación GIS y *Troubleshooting* de errores comunes. |
| **[🧪 Calidad y experimentos](quality-and-experiments.md)** | Explicación de la suite de 76 pruebas de regresión, auditoría de hash SHA-256 del checkpoint, y la metodología experimental para comparar métricas. |
| **[🛠️ Desarrollo y despliegue](development-and-deployment.md)** | Flujo seguro de cambios, justificación de decisiones técnicas, límites del proyecto (ausencia de soporte en la nube) y trabajo futuro. |

---

## 🧭 Recorridos Recomendados

Dependiendo de tu rol en el proyecto, te recomendamos los siguientes recorridos rápidos:

### 👤 Soy un usuario operativo o evaluador
Si necesitás correr el proyecto y probar la simulación:
1. [Instalación y primera ejecución](getting-started.md)
2. [Carga de pedidos y formatos Excel](user-manual.md#2-pedidos)
3. [Navegación del Dashboard y vistas GIS](user-manual.md#dashboard)
4. [Solución de problemas comunes](user-manual.md#solución-de-problemas)

### 👨‍💻 Soy un desarrollador nuevo en el repo
Si clonaste el repositorio y tenés que mantenerlo o extenderlo:
1. [Arquitectura y responsabilidades](system-design.md#arquitectura)
2. [Flujo seguro de desarrollo (CONTRIBUTING)](../CONTRIBUTING.md)
3. [Pruebas automatizadas](quality-and-experiments.md#regresión)
4. [Límites de despliegue](development-and-deployment.md#despliegue)

### 🧠 Quiero analizar el Aprendizaje por Refuerzo (RL)
Si te interesa el núcleo algorítmico del planificador:
1. [RL como propuesta central](system-design.md#rl-como-propuesta-central)
2. [El método Híbrido](system-design.md#híbrido)
3. [Checkpoint productivo y máscara temporal](system-design.md#rl-productivo)
4. [Metodología para comparar algoritmos](quality-and-experiments.md#metodología-experimental)
