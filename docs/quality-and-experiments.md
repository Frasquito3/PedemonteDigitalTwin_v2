<div align="center">

# Calidad y experimentos

<a href="../README.md">🏠 Inicio</a> ·
<a href="README.md">📚 Documentación</a> ·
<a href="getting-started.md">🚀 Comenzar</a> ·
<a href="system-design.md">🏗️ Sistema</a> ·
<a href="user-manual.md">🎛️ Operación</a> ·
<strong>🧪 Calidad</strong> ·
<a href="development-and-deployment.md">🛠️ Desarrollo</a>

</div>

---

## Estado validado

<p>
  <img alt="Tests" src="https://img.shields.io/badge/tests-76%2F76-2ea44f?style=flat-square">
  <img alt="Audit" src="https://img.shields.io/badge/auditoría-55%2F55-00897B?style=flat-square">
  <img alt="GIS" src="https://img.shields.io/badge/GIS-1%3A10000-1565C0?style=flat-square">
</p>

Checkpoint esperado:

```text
7d2838963f578656919822ee258e9c87c38257829ea801bbd3b41fcf26d20da4
```

## Regresión

Desde `python/`:

```powershell
$py = (Resolve-Path "..\.venv\Scripts\python.exe").Path

& $py -m compileall -q planner scripts tests
& $py -m pytest -q --tb=short
```

Resultado:

```text
76 passed
```

### Cobertura

| Área | Cobertura funcional |
|---|---|
| Algoritmos | factibilidad, reproducibilidad, GA, HÍBRIDO |
| Dominio | split, capacidad, volcador |
| Integración | codecs, selector, Excel, caché |
| RL | entorno, hash, recompensa, currículo |
| Routing | recursos, espera, tardanza, objetivo |

## Smoke tests AnyLogic

### Cambio visual

- Build Model.
- Importación Excel.
- Preparación.
- Ejecución RL.
- General, Camión 0 y Camión 1.
- Textos sin recortes ni identificadores internos.

### Cambio de integración o planificación

- GREEDY.
- RANDOM.
- GA.
- RL.
- HÍBRIDO.
- Métricas finales.
- Consola sin errores.
- Git limpio.

## Qué invalida resultados previos

> [!WARNING]
> Una comparación deja de ser equivalente cuando cambia cualquiera de estos elementos:

- costos;
- caché;
- tráfico;
- capacidad;
- cantidad de camiones;
- split;
- checkpoint;
- máscara;
- semillas;
- configuración operativa.

## Metodología experimental

### Objetivo

Comparar:

```text
RL
HIBRIDO
GREEDY
RANDOM
GA
```

RL se presenta primero como propuesta central. Los demás métodos son referencias y mecanismos de validación.

### Condiciones comunes

Cada corrida debe conservar:

- pedidos;
- fecha;
- turno;
- caché;
- configuración;
- semilla de escenario;
- semilla de ejecución.

### Segmentos

```text
3–5 tareas
6–8 tareas
9–10 tareas
11–12 tareas
```

Incluir casos con:

- ventanas;
- volcador;
- split;
- distancias variadas;
- riesgo temporal.

### Métricas

| Métrica | Fuente |
|---|---|
| Factibilidad | Validador |
| Costo estimado | Python |
| Costo real | AnyLogic |
| Distancia | AnyLogic |
| Duración | AnyLogic |
| Viajes | Plan y simulación |
| Pedidos tardíos | Estimación y simulación |
| Tardanza | Estimación y simulación |
| Desbalance | Objetivo y simulación |
| Tiempo de planificación | Selector Python |

### Repeticiones sugeridas

```text
10 instancias por segmento
5 semillas para RANDOM y GA
```

RL y GREEDY pueden ejecutarse una vez por instancia si mantienen determinismo. La semilla GA de HÍBRIDO debe quedar registrada.

### Formato de salida

```text
instance_id
segment
method
scenario_seed
execution_seed
planner_seed
feasible
estimated_cost
real_cost
distance_km
operation_minutes
trips
late_orders
tardiness_minutes
finish_imbalance_minutes
planning_time_ms
```

### Análisis

Informar:

- media;
- mediana;
- desviación estándar;
- mínimo;
- máximo;
- porcentaje factible;
- diferencia contra RL;
- diferencia contra GREEDY;
- mejora de HÍBRIDO sobre su semilla RL.

> [!IMPORTANT]
> No ocultar casos donde RL sea superado. La discusión debe considerar calidad, velocidad, estabilidad, factibilidad y límites.

## Reproducibilidad

Registrar siempre:

```text
commit
hash del checkpoint
versión de caché
configuración
fecha
entorno Python
semillas
```

## Evidencia

`python/results/` no se versiona. Los resultados importantes deben transformarse en:

- una tabla consolidada;
- documentación estable;
- pruebas;
- manifiestos;
- hashes.

---

<div align="center">

<a href="user-manual.md">← Manual de uso</a>
&nbsp;·&nbsp;
<a href="README.md">Índice</a>
&nbsp;·&nbsp;
<a href="development-and-deployment.md">Desarrollo y despliegue →</a>

</div>
