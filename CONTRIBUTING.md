<div align="center">

# Contribuir a Pedemonte Digital Twin

[🏠 Inicio](README.md) •
[📚 Documentación](docs/README.md) •
[🛠️ Desarrollo completo](docs/development-and-deployment.md)

</div>

---

> [!CAUTION]
> El repositorio combina el modelo de simulación de AnyLogic, el código Python, cachés viales y un checkpoint validado de RL. **Un cambio en cualquiera de estos elementos puede alterar la reproducibilidad global y romper los 76 tests de regresión**. Procedé con cuidado.

## 🌊 Flujo de Trabajo Recomendado

1. **Mantener limpio el repositorio**: Asegurá que `git status --short` esté vacío antes de empezar.
2. **Implementar cambios de manera aislada**: Sea en el `.alp` de AnyLogic o en los algoritmos de Python.
3. **Validación local**: 
   - En Python: correr pytest y pyrefly.
   - En AnyLogic: ejecutar un *Smoke Test* visual (Build Model, Importar Excel, Ejecutar RL y Híbrido).
4. **Commit semántico**: Usar *Conventional Commits*.

## 📝 Convención de Commits

Usamos [Conventional Commits](https://www.conventionalcommits.org/) con títulos en **inglés**. No incluyas números de fase ni etiquetas como "final".

**Tipos permitidos**:
- `feat:` Nuevas características.
- `fix:` Solución de errores.
- `refactor:` Mejoras de código sin alterar funcionalidad.
- `docs:` Cambios en la documentación (como este).
- `test:` Inclusión o mejora de pruebas.
- `chore:` Mantenimiento (dependencias, configuraciones).

*Ejemplo:* `feat: add GIS truck follow controls`

## 🐍 Cambios en Python

Cualquier nuevo método de planificación o modificación en los existentes debe garantizar:
1. Devolver un objeto `PlanTurno` válido.
2. Respetar las reglas de dominio: capacidad, split automático, ventanas de tiempo y regla de volcador.
3. Utilizar el proveedor vial inyectado.
4. Evaluarse bajo la función objetivo común del sistema.

Para validar cambios en Python (desde la carpeta `python/`):
```powershell
$py = (Resolve-Path "..\.venv\Scripts\python.exe").Path
& $py -m compileall -q planner scripts tests
& $py -m pytest -q --tb=short
```

## ⚙️ Cambios en AnyLogic

Antes de realizar el commit de cambios en AnyLogic:
1. Asegurá que compila correctamente (`Build Model`).
2. Probá el recorrido crítico: cargar Excel, preparar, generar plan y seguir los camiones en GIS.
3. **Importante**: Revisá el diff usando `git diff --stat`. 
> [!WARNING]
> La variable `pyCommunicator.pythonExecPath` almacena tu ruta local absoluta. **No incluyas cambios a esta ruta en tu commit** mezclados con cambios funcionales.

## 💾 Políticas RL y Caché Vial

- **Checkpoint RL (`python/models/rl/`)**: Los entrenamientos guardados en `rl_artifacts/` **no reemplazan** automáticamente el modelo productivo. La promoción de un nuevo checkpoint requiere validación manual, actualización del hash SHA-256 en la documentación y pasar todas las pruebas.
- **Caché Vial (`python/data/routing/cache_vial.csv`)**: La caché es **estricta**. Si se necesita un tramo vial nuevo, debés agregarlo a la caché y correr la regresión. No se permite habilitar *fallbacks* dinámicos en producción.

## 🚫 Archivos NO Versionables

Asegurate de que los siguientes archivos o carpetas no entren en el control de versiones:
- `.venv/`
- `__pycache__/` y `.pytest_cache/`
- `python/results/` (resultados de pruebas experimentales)
- `python/rl_artifacts/` (checkpoints candidatos no promovidos)
- `anylogic/**/al_uploads/`
- Planillas `.xlsx` personales o copias de seguridad del modelo `.alp`.
