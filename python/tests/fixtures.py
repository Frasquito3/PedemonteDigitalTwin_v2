from planner.schema import (
    InstanciaTurno,
    PedidoInput,
    Turno,
)


def crear_instancia_demo() -> InstanciaTurno:
    lat_corralon = -32.8495006
    lon_corralon = -60.722653

    pedidos = [
        PedidoInput(
            pedido_id="P001",
            pedido_original_id="P001",
            numero_parte=1,
            total_partes=1,
            turno=Turno.MANANA,
            latitud=lat_corralon + 0.010,
            longitud=lon_corralon + 0.010,
            unidades_capacidad=3,
            requiere_volcador=False,
            tiene_ventana_especifica=False,
            hora_desde_min=450,
            hora_hasta_min=720,
        ),

        PedidoInput(
            pedido_id="P002",
            pedido_original_id="P002",
            numero_parte=1,
            total_partes=1,
            turno=Turno.MANANA,
            latitud=lat_corralon + 0.015,
            longitud=lon_corralon + 0.005,
            unidades_capacidad=4,
            requiere_volcador=False,
            tiene_ventana_especifica=False,
            hora_desde_min=450,
            hora_hasta_min=720,
        ),

        PedidoInput(
            pedido_id="P003",
            pedido_original_id="P003",
            numero_parte=1,
            total_partes=1,
            turno=Turno.MANANA,
            latitud=lat_corralon - 0.010,
            longitud=lon_corralon + 0.020,
            unidades_capacidad=2,
            requiere_volcador=True,
            tiene_ventana_especifica=False,
            hora_desde_min=450,
            hora_hasta_min=720,
        ),

        PedidoInput(
            pedido_id="P004",
            pedido_original_id="P004",
            numero_parte=1,
            total_partes=1,
            turno=Turno.MANANA,
            latitud=lat_corralon - 0.015,
            longitud=lon_corralon - 0.010,
            unidades_capacidad=6,
            requiere_volcador=False,
            tiene_ventana_especifica=False,
            hora_desde_min=450,
            hora_hasta_min=720,
        ),
    ]

    return InstanciaTurno(
        instancia_id="FASE6-TEST-001",

        fecha_operacion="2026-08-13",

        turno=Turno.MANANA,

        pedidos=pedidos,

        lat_corralon=lat_corralon,

        lon_corralon=lon_corralon,

        capacidad_camion=8,

        cantidad_camiones=2,

        hora_inicio_turno_min=450,

        hora_fin_objetivo_min=720,

        hora_fin_tolerancia_min=735,

        seed_escenario=1001,

        seed_ejecucion=2001,
    )