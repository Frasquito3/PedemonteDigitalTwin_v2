<div align="center">

# 🛠️ Desarrollo, Decisiones y Despliegue

[🏠 Inicio](../README.md) •
[📚 Documentación](README.md) •
[🚀 Comenzar](getting-started.md) •
[🏗️ Sistema](system-design.md) •
[🎛️ Operación](user-manual.md) •
[🧪 Calidad](quality-and-experiments.md) •
**🛠️ Desarrollo**

</div>

---

En esta sección se explican las razones técnicas detrás del diseño actual del gemelo digital y se definen los parámetros de su posible evolución y soporte futuro.

## 📝 Decisiones Arquitectónicas

1. **Python Local + AnyLogic de Escritorio (Pypeline)**
   *Decisión:* El sistema es totalmente monolítico en escritorio. Esto simplifica enormemente la evaluación local del aprendizaje por refuerzo, ya que no se incurre en latencias de HTTP para el validador, y no se lidia con persistencia distribuida del estado.
2. **Máscara Temporal Dura (RL)**
   *Decisión:* Se rechazó una estrategia de recompensas negativas masivas (Soft Mask) a favor de un bloqueo algorítmico estricto en la matriz de decisión (Hard Mask). Esto obliga al RL a proponer movimientos únicamente factibles, eliminando planes imposibles rápidamente y disminuyendo drásticamente el espacio de búsqueda inútil.
3. **Caché Vial Estricta y No-Fallback**
   *Decisión:* Se descartó utilizar la API de ruteo local de AnyLogic u OpenStreetMap dinámico para evitar la contaminación de métricas. Todas las simulaciones utilizan la *exacta misma* distancia en metros guardada en el CSV, logrando 100% de reproducibilidad entre ejecuciones.

---

## 🚫 Despliegue: AnyLogic Cloud

> [!CAUTION]
> El proyecto actual **NO ES COMPATIBLE** con AnyLogic Cloud y no debe intentarse subirlo como un producto productivo en ese entorno.

**¿Por qué?**
Pypeline, el puente que une AnyLogic con Python, se apoya en un binario ejecutable (`python.exe`) con acceso local a discos, modelos y librerías C (como Numpy). AnyLogic Cloud opera en un entorno web que bloquea ejecuciones binarias por seguridad y carece de la topología local (las carpetas `/models/rl/` o `/data/routing/`). 

### Posibles Alternativas Futuras
Para llegar a la nube, se requiere un refactor completo a un esquema de microservicios:
- Un backend (FastAPI o Flask) donde se aloje el código Python, el validador y el pipeline del modelo RL, que exponga endpoints REST.
- El modelo AnyLogic Cloud haría peticiones HTTP hacia estos endpoints.
- Las penalizaciones por red y latencias obligarían a diseñar procesos asíncronos en el código Java.

---

## 🔮 Trabajo Futuro (Roadmap Posible)

Los siguientes componentes están definidos y estudiados, pero **permanecen fuera del alcance temporal productivo**:

- **Plugin HTML (App Web):** Una interfaz de formulario web donde el cliente local no necesite abrir AnyLogic. Esta UI consumiría un archivo y delegaría a la lógica local.
- **Vista Urbana 3D:** Renderizado tridimensional de la ciudad, con edificios representativos y los camiones en escala real. Aunque es vistoso, el costo computacional de AnyLogic 3D interfiere en las simulaciones repetitivas y fue descartado de la versión principal 2D por eficiencia.
- **Tercer Camión y Segmentación Dinámica:** Extender el simulador para que el planificador pueda incorporar dinámicamente el "Tercer Camión de Respaldo", lo que requerirá un re-entrenamiento del entorno RL.
