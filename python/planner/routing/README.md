# Módulo `planner.routing` — Modelado Espacial y Estimación de Rutas

El submódulo `routing` transforma asignaciones abstractas de pedidos en itinerarios físicos sobre la red vial, calculando distancias, duraciones y costos multiobjetivo.

---

## Archivos del Módulo

### `decoder.py`
Decodificación determinista de soluciones:
- `decodificar_plan_permutacion`: Transforma una permutación de pedidos en viajes respetando capacidades, volcador y asignando cada viaje al camión con menor tiempo acumulado.
- Reconstrucción de trazas horarias con tiempos de carga en planta, viaje de ida, descarga en cliente y retorno.

### `objective.py`
Función objetivo multiobjetivo unificada:
- `evaluar_plan_estimado`: Calcula la métrica escalar de costo integrando distancia recorrida (combustible), tiempo de operación, penalización por minutos de tardanza y penalización por desbalance horario entre camiones.

### `travel.py`
Cálculo de matrices de distancia y duración:
- `ProveedorHaversineAjustado`: Estimación geodésica calibrada con factor de sinuosidad urbana.
- `construir_matriz_viaje`: Construcción eficiente de matrices completas punto a punto entre el corralón y los destinos.

### `vial_cache.py`
Persistencia y optimización de consultas geográficas:
- `ProveedorVialCachePersistente`: Consulta distancias en un archivo CSV precalculado en disco (`cache_vial.csv`), evitando latencias de red y garantizando determinismo absoluto.

### `operations.py`
Constantes operativas del depósito, velocidades medias de camiones y tiempos fijos de carga/descarga.

