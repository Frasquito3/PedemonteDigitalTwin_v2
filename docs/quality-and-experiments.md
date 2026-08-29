<div align="center">

# 🧪 Calidad y Metodología Experimental

[🏠 Inicio](../README.md) •
[📚 Documentación](README.md) •
[🚀 Comenzar](getting-started.md) •
[🏗️ Sistema](system-design.md) •
[🎛️ Operación](user-manual.md) •
**🧪 Calidad** •
[🛠️ Desarrollo](development-and-deployment.md)

</div>

---

Pedemonte Digital Twin basa su fiabilidad en la rigidez de su validación local y reproducibilidad técnica. Esta sección explica cómo mantener esta calidad operativa y bajo qué métricas se evalúan los distintos métodos de planificación.

## ✅ Suite de Regresión

El proyecto cuenta con un paquete estricto de pruebas automatizadas. Cualquier ajuste en Python debe ejecutarlas con éxito.

```powershell
Set-Location .\python
$py = (Resolve-Path "..\.venv\Scripts\python.exe").Path
& $py -m pytest -q --tb=short
```

> **Resultado Garantizado:** 76/76 pruebas pasadas correctamente. 
Las pruebas cubren la factibilidad del Validador, las transformaciones y *splits*, las restricciones horarias y el objetivo compartido.

## 🔐 Checkpoint RL y Reproducibilidad

Todo modelo desplegado en producción debe corresponder a un entrenamiento específico (`pedemonte-rl-single-policy-v1`). Para evitar corrupciones locales, se provee el hash `SHA-256` exacto del archivo `python/models/rl/rl_policy.zip`.

```text
7D2838963F578656919822EE258E9C87C38257829EA801BBD3B41FCF26D20DA4
```
**Importante:** Alterar semillas, la configuración del modelo operativo de AnyLogic, el peso de penalizaciones, o el tamaño de la capacidad de camiones, rompe la validación de la política existente, requiriendo un re-entrenamiento exhaustivo.

## 🔬 Metodología de Experimentación y Comparación

Si vas a agregar un método o re-evaluar la política actual, debés utilizar el modelo estricto de comparación.

### Alcance Demostrado y Límites
La política actual está comprobada de forma productiva para escenarios de **hasta 12 tareas operativas** por turno. El sistema es totalmente capaz de escalar (hasta las 30 soportadas por el puente), pero los tiempos de inferencia y la tasa de entregas fallidas aumentarán progresivamente.

### Segmentación de Tests Recomendada
- Nivel Inicial: 3 a 5 tareas.
- Nivel Medio: 6 a 8 tareas.
- Nivel Alto: 9 a 10 tareas.
- Nivel de Estrés Verificado: 11 a 12 tareas.

### Transparencia de Métricas
Al tabular los resultados entre `RL`, `Híbrido`, `Greedy`, `Random` y `GA`, los informes **deben ser transparentes** e incluir, por método:
- **Tasa de factibilidad** inicial.
- **Costo Operativo Real** (Tiempos de AnyLogic + Penalizaciones de Python).
- **Tiempo de Planificación** (En milisegundos).
- **Distancias y Tardanzas** absolutas.

> [!IMPORTANT]
> Nunca ocultes un escenario en el que `Greedy` o `Híbrido` superen al `RL` base. El RL es altamente exploratorio y su tiempo de inferencia instantáneo es valioso, pero en ciertas ventanas rígidas, un determinista local (`Greedy`) puede entregar un costo menor de forma matemática. Esto es esperable y debe quedar registrado.
