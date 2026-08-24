from __future__ import annotations

from random import Random

from planner.core.schema import InstanciaTurno, PedidoInput, Turno
from planner.domain.validator import validar_instancia
from planner.evaluation.classic_instances import (
    COORDENADA_CERCANA,
    COORDENADA_ESTE,
    COORDENADA_NORTE,
    LAT_CORRALON,
    LON_CORRALON,
    CasoBenchmarkClasico,
)


ESTRATOS_STRESS_RL = (
    "MIXTO",
    "VENTANAS",
    "VOLCADOR",
    "SPLIT",
    "COMBINADO",
)

_COORDENADAS = (
    COORDENADA_NORTE,
    COORDENADA_ESTE,
    COORDENADA_CERCANA,
)


def _pedido(
    pedido_id: str,
    *,
    coordenada: tuple[float, float],
    unidades: int,
    requiere_volcador: bool = False,
    tiene_ventana_especifica: bool = False,
    hora_desde_min: int = 450,
    hora_hasta_min: int = 720,
    pedido_original_id: str | None = None,
    numero_parte: int = 1,
    total_partes: int = 1,
) -> PedidoInput:
    return PedidoInput(
        pedido_id=pedido_id,
        pedido_original_id=(
            pedido_original_id
            if pedido_original_id is not None
            else pedido_id
        ),
        numero_parte=numero_parte,
        total_partes=total_partes,
        turno=Turno.MANANA,
        latitud=coordenada[0],
        longitud=coordenada[1],
        unidades_capacidad=unidades,
        requiere_volcador=requiere_volcador,
        tiene_ventana_especifica=tiene_ventana_especifica,
        hora_desde_min=hora_desde_min,
        hora_hasta_min=hora_hasta_min,
        cliente=f"Cliente {pedido_id}",
        direccion=f"Dirección stress {pedido_id}",
        barrio="Stress vial controlado",
        observaciones="Fase 15R.6B",
    )


def _ventana(rng: Random, indice: int) -> tuple[int, int]:
    aperturas = (465, 495, 525, 555, 585, 615)
    apertura = aperturas[(indice + rng.randrange(len(aperturas))) % len(aperturas)]
    duracion = rng.choice((45, 60, 75))
    cierre = min(apertura + duracion, 705)
    if cierre <= apertura:
        cierre = apertura + 30
    return apertura, cierre


def _id_pedido(
    estrato: str,
    repeticion: int,
    indice: int,
    rng: Random,
) -> str:
    token = rng.randrange(10_000, 99_999)
    return f"S{estrato[:3]}-{repeticion:02d}-{indice:02d}-{token}"


def _crear_pedidos_mixto(
    rng: Random,
    repeticion: int,
) -> list[PedidoInput]:
    cantidad = rng.randint(4, 8)
    pedidos = [
        _pedido(
            _id_pedido("MIXTO", repeticion, indice, rng),
            coordenada=rng.choice(_COORDENADAS),
            unidades=rng.randint(1, 8),
        )
        for indice in range(1, cantidad + 1)
    ]
    rng.shuffle(pedidos)
    return pedidos


def _crear_pedidos_ventanas(
    rng: Random,
    repeticion: int,
) -> list[PedidoInput]:
    cantidad = rng.randint(4, 7)
    pedidos: list[PedidoInput] = []
    for indice in range(1, cantidad + 1):
        desde, hasta = _ventana(rng, indice)
        pedidos.append(
            _pedido(
                _id_pedido("VENTANAS", repeticion, indice, rng),
                coordenada=rng.choice(_COORDENADAS),
                unidades=rng.randint(1, 4),
                tiene_ventana_especifica=True,
                hora_desde_min=desde,
                hora_hasta_min=hasta,
            )
        )
    rng.shuffle(pedidos)
    return pedidos


def _crear_pedidos_volcador(
    rng: Random,
    repeticion: int,
) -> list[PedidoInput]:
    cantidad = rng.randint(4, 7)
    cantidad_volcadores = 1 if cantidad <= 5 else rng.choice((1, 2))
    indices_volcador = set(
        rng.sample(range(cantidad), k=cantidad_volcadores)
    )
    pedidos = [
        _pedido(
            _id_pedido("VOLCADOR", repeticion, indice + 1, rng),
            coordenada=rng.choice(_COORDENADAS),
            unidades=rng.randint(2, 6),
            requiere_volcador=indice in indices_volcador,
        )
        for indice in range(cantidad)
    ]
    rng.shuffle(pedidos)
    return pedidos


def _crear_pedidos_split(
    rng: Random,
    repeticion: int,
) -> list[PedidoInput]:
    original_id = f"SSPL-{repeticion:02d}-ORIGINAL"
    coordenada_split = rng.choice(_COORDENADAS)
    segunda_parte = rng.randint(2, 7)
    pedidos = [
        _pedido(
            f"{original_id}-P1",
            pedido_original_id=original_id,
            numero_parte=1,
            total_partes=2,
            coordenada=coordenada_split,
            unidades=8,
        ),
        _pedido(
            f"{original_id}-P2",
            pedido_original_id=original_id,
            numero_parte=2,
            total_partes=2,
            coordenada=coordenada_split,
            unidades=segunda_parte,
        ),
    ]
    cantidad_normales = rng.randint(2, 5)
    for indice in range(1, cantidad_normales + 1):
        pedidos.append(
            _pedido(
                _id_pedido("SPLIT", repeticion, indice, rng),
                coordenada=rng.choice(_COORDENADAS),
                unidades=rng.randint(1, 6),
            )
        )
    rng.shuffle(pedidos)
    return pedidos


def _crear_pedidos_combinado(
    rng: Random,
    repeticion: int,
) -> list[PedidoInput]:
    original_id = f"SCOM-{repeticion:02d}-ORIGINAL"
    coordenada_split = rng.choice(_COORDENADAS)
    desde_split, hasta_split = _ventana(rng, 1)
    pedidos: list[PedidoInput] = [
        _pedido(
            f"{original_id}-P1",
            pedido_original_id=original_id,
            numero_parte=1,
            total_partes=2,
            coordenada=coordenada_split,
            unidades=8,
            tiene_ventana_especifica=True,
            hora_desde_min=desde_split,
            hora_hasta_min=hasta_split,
        ),
        _pedido(
            f"{original_id}-P2",
            pedido_original_id=original_id,
            numero_parte=2,
            total_partes=2,
            coordenada=coordenada_split,
            unidades=rng.randint(2, 6),
            tiene_ventana_especifica=True,
            hora_desde_min=desde_split,
            hora_hasta_min=hasta_split,
        ),
    ]

    cantidad_extra = rng.randint(3, 5)
    indice_volcador = rng.randrange(cantidad_extra)
    for indice in range(cantidad_extra):
        tiene_ventana = rng.random() < 0.70
        desde, hasta = _ventana(rng, indice + 2)
        pedidos.append(
            _pedido(
                _id_pedido("COMBINADO", repeticion, indice + 1, rng),
                coordenada=rng.choice(_COORDENADAS),
                unidades=rng.randint(1, 6),
                requiere_volcador=indice == indice_volcador,
                tiene_ventana_especifica=tiene_ventana,
                hora_desde_min=desde if tiene_ventana else 450,
                hora_hasta_min=hasta if tiene_ventana else 720,
            )
        )
    rng.shuffle(pedidos)
    return pedidos


_CREADORES = {
    "MIXTO": _crear_pedidos_mixto,
    "VENTANAS": _crear_pedidos_ventanas,
    "VOLCADOR": _crear_pedidos_volcador,
    "SPLIT": _crear_pedidos_split,
    "COMBINADO": _crear_pedidos_combinado,
}


def crear_casos_stress_rl(
    cantidad_por_estrato: int = 12,
    *,
    seed_base: int = 16_600,
) -> tuple[CasoBenchmarkClasico, ...]:
    if cantidad_por_estrato <= 0:
        raise ValueError("cantidad_por_estrato debe ser > 0.")

    casos: list[CasoBenchmarkClasico] = []
    for indice_estrato, estrato in enumerate(ESTRATOS_STRESS_RL):
        creador = _CREADORES[estrato]
        for repeticion in range(1, cantidad_por_estrato + 1):
            seed = seed_base + indice_estrato * 1_000 + repeticion
            rng = Random(seed)
            pedidos = creador(rng, repeticion)
            instancia = InstanciaTurno(
                instancia_id=f"STRESS-{estrato}-{repeticion:02d}",
                fecha_operacion="2026-08-24",
                turno=Turno.MANANA,
                pedidos=pedidos,
                lat_corralon=LAT_CORRALON,
                lon_corralon=LON_CORRALON,
                capacidad_camion=8,
                cantidad_camiones=2,
                hora_inicio_turno_min=450,
                hora_fin_objetivo_min=720,
                hora_fin_tolerancia_min=735,
                seed_escenario=seed,
                seed_ejecucion=seed + 10_000,
            )
            errores = validar_instancia(instancia)
            if errores:
                raise RuntimeError(
                    f"Instancia stress inválida {instancia.instancia_id}: "
                    + " | ".join(errores)
                )
            casos.append(
                CasoBenchmarkClasico(
                    caso_id=instancia.instancia_id,
                    categoria=estrato,
                    descripcion=(
                        "Instancia stress determinística sobre el conjunto "
                        "de nodos de la caché vial validada."
                    ),
                    instancia=instancia,
                )
            )

    return tuple(casos)
