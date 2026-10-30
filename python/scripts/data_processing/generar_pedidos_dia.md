| Atributo | Método de generación | Justificación |
| :--- | :--- | :--- |
| **Cantidad de pedidos (turno)** | Muestreo sobre registros reales (`cant_pedidos_turnos.csv`) con variación aleatoria de ±1 pedido. | Respeta la distribución real de demanda del corralón sin repetir valores fijos. |
| `pedido_id` | Secuencial incremental por turno (`PED_001`, `PED_002`, ...). | Identificador unívoco requerido para trazabilidad en algoritmos y AnyLogic. |
| `turno` | Asignación fija del bloque horario de trabajo (`MANANA` o `TARDE`). | Modela la operación partida en dos turnos diarios independientes. |
| `latitud`, `longitud` | Selección de punto histórico ponderada por demanda barrial (`probabilidades_demanda.csv`) con desplazamiento aleatorio de 40 a 145 m. | Conserva la distribución espacial real de entregas sin duplicar coordenadas exactas. |
| `unidades` | Reparto del volumen del turno (~24 unidades / 3 camiones de 8 slots) en enteros ≥ 1. | Respeta la capacidad física de los camiones y el promedio real de despacho diario. |
| `requiere_volcador` | Asignación binaria ("SI" / "NO") con probabilidad de 1 cada 8 turnos (p = 0.125). | Refleja la baja frecuencia operativa real de este vehículo (~1 salida cada 4 días). |
| `tiene_ventana`, `hora_desde`, `hora_hasta` | Franjas en múltiplos de 15 min según distribución empírica: sin ventana (45%), antes de X (25%), después de X (22%) o entre X e Y (8%). | Emula los patrones y tipos de restricciones horarias solicitadas habitualmente por los clientes. |
