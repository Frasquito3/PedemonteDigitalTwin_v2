<div align="center">

# Centro de Documentación Técnica

<p>
  <img alt="Guías" src="https://img.shields.io/badge/Documentaci%C3%B3n-5_Gu%C3%ADas_T%C3%A9cnicas-1565C0?style=for-the-badge">
</p>

[Inicio](../README.md) •
**Documentación** •
[Primeros Pasos](getting-started.md) •
[Diseño del Sistema](system-design.md) •
[Manual de Operación](user-manual.md) •
[Calidad y Experimentos](quality-and-experiments.md) •
[Desarrollo y Despliegue](development-and-deployment.md)

</div>

---

La documentación de **Pedemonte Digital Twin** está estructurada para cubrir tanto la operación funcional del gemelo digital como el diseño algorítmico y el mantenimiento del código fuente. El código, la suite de pruebas automatizadas y los manifiestos criptográficos (`sha256`) constituyen la fuente de verdad ejecutable del sistema.

Adicionalmente, la referencia detallada de paquetes, módulos, clases y funciones de Python (`planner` y `scripts`) se compila automáticamente mediante `pdoc` utilizando las plantillas definidas en `docs/pdoc_template/`.

## Mapa Documental

| Documento | Alcance y Contenido |
| :--- | :--- |
| **[Primeros pasos](getting-started.md)** | Requisitos del sistema, creación del entorno virtual Python 3.11, instalación de librerías en AnyLogic (`AnyLogic-Pypeline` y `AnyLogicWebToolkit`), verificación de integridad y primera corrida. |
| **[Diseño del sistema](system-design.md)** | Arquitectura desacoplada AnyLogic–Python, formulación de los 5 motores de planificación (`RL`, `HIBRIDO`, `GA`, `GREEDY`, `RANDOM`), función objetivo compartida, caché vial estricta y pipeline de demanda histórica. |
| **[Manual de operación](user-manual.md)** | Uso del formulario web de pedidos, formato de planillas Excel (`.xlsx`), configuración de turnos y semillas, operación del panel de simulación, vistas GIS y resolución de problemas frecuentes. |
| **[Calidad y experimentos](quality-and-experiments.md)** | Ejecución de las 174 pruebas automatizadas, verificación de tipos con `pyrefly`, control de estilo con `ruff`, auditoría del checkpoint RL productivo y metodología de comparación experimental con Alpyne. |
| **[Desarrollo y despliegue](development-and-deployment.md)** | Decisiones arquitectónicas, flujo de trabajo con hooks de Git (`.githooks`), integración continua en GitHub Actions, restricciones respecto a AnyLogic Cloud y líneas de evolución futura. |

---

## Recorridos de Lectura Sugeridos

### Perfil Operativo o Evaluador Funcional
Para configurar el entorno y ejecutar pruebas de despacho en el simulador:
1. [Instalación y configuración del entorno](getting-started.md#instalacion-y-configuracion)
2. [Carga de pedidos y formato Excel](user-manual.md#2-carga-y-generacion-de-pedidos)
3. [Seguimiento en el panel y vistas GIS](user-manual.md#4-visualizacion-en-el-panel-y-gis)
4. [Resolución de problemas frecuentes](user-manual.md#5-resolucion-de-problemas-troubleshooting)

### Perfil de Desarrollo y Mantenimiento
Para extender los módulos de planificación, modificar pipelines o contribuir al repositorio:
1. [Arquitectura general y responsabilidades](system-design.md#arquitectura-general)
2. [Estándares de código, hooks y CI/CD](development-and-deployment.md#flujo-de-desarrollo-y-calidad-de-codigo)
3. [Suite de regresión y verificación estática](quality-and-experiments.md#suite-de-regresion-y-verificacion-estatica)
4. [Restricciones de despliegue](development-and-deployment.md#restricciones-de-despliegue-anylogic-cloud)

### Perfil de Investigación Algorítmica y Aprendizaje por Refuerzo
Para analizar el diseño de los optimizadores y reproducir los experimentos comparativos:
1. [Algoritmos de planificación y política RL](system-design.md#algoritmos-de-planificacion)
2. [Función objetivo compartida y modelo operativo](system-design.md#funcion-objetivo-compartida)
3. [Checkpoint RL y validación curricular](quality-and-experiments.md#checkpoint-rl-y-reproducibilidad)
4. [Metodología de experimentación y comparación](quality-and-experiments.md#metodologia-de-experimentacion-y-comparacion)
