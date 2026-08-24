from __future__ import annotations

from dataclasses import dataclass

from planner.core.schema import (
    InstanciaTurno,
    PedidoInput,
    Turno,
)


LAT_CORRALON = -32.8495006
LON_CORRALON = -60.722653

COORDENADA_NORTE = (-32.8310000, -60.7190000)
COORDENADA_ESTE = (-32.8595006, -60.7026530)
COORDENADA_CERCANA = (-32.8410000, -60.7210000)


@dataclass(frozen=True)
class CasoBenchmarkClasico:
    caso_id: str
    categoria: str
    descripcion: str
    instancia: InstanciaTurno


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
    cliente: str = "",
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
        tiene_ventana_especifica=(
            tiene_ventana_especifica
        ),
        hora_desde_min=hora_desde_min,
        hora_hasta_min=hora_hasta_min,
        cliente=cliente or pedido_id,
        direccion=f"Dirección de prueba {pedido_id}",
        barrio="Benchmark vial",
        observaciones="Caso reproducible Fase 15R.5A",
    )


def _instancia(
    instancia_id: str,
    pedidos: list[PedidoInput],
    *,
    seed: int,
) -> InstanciaTurno:
    return InstanciaTurno(
        instancia_id=instancia_id,
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


def crear_casos_benchmark_clasico(
) -> tuple[CasoBenchmarkClasico, ...]:
    """
    Crea una batería estable de instancias pequeñas y auditables.

    Todas las ubicaciones pertenecen al conjunto vial validado en las
    Fases 15R.3H–15R.4F. Los casos de split reutilizan coordenadas, por
    lo que no requieren tramos viales adicionales.
    """
    simple = CasoBenchmarkClasico(
        caso_id="B01_SIMPLE",
        categoria="SIMPLE",
        descripcion=(
            "Dos pedidos pequeños que caben en un único viaje."
        ),
        instancia=_instancia(
            "BENCH-B01-SIMPLE",
            [
                _pedido(
                    "B01-NORTE",
                    coordenada=COORDENADA_NORTE,
                    unidades=1,
                ),
                _pedido(
                    "B01-ESTE",
                    coordenada=COORDENADA_ESTE,
                    unidades=1,
                ),
            ],
            seed=15_101,
        ),
    )

    carga_paralela = CasoBenchmarkClasico(
        caso_id="B02_CARGA_PARALELA",
        categoria="CARGA_PARALELA",
        descripcion=(
            "Dos pedidos de capacidad completa; obliga a usar los "
            "dos camiones desde el comienzo."
        ),
        instancia=_instancia(
            "BENCH-B02-CARGA-PARALELA",
            [
                _pedido(
                    "B02-NORTE-8",
                    coordenada=COORDENADA_NORTE,
                    unidades=8,
                ),
                _pedido(
                    "B02-ESTE-8",
                    coordenada=COORDENADA_ESTE,
                    unidades=8,
                ),
            ],
            seed=15_102,
        ),
    )

    multiviaje = CasoBenchmarkClasico(
        caso_id="B03_MULTIVIAJE",
        categoria="MULTIVIAJE",
        descripcion=(
            "Tres pedidos de capacidad completa; uno de los camiones "
            "debe realizar un segundo viaje."
        ),
        instancia=_instancia(
            "BENCH-B03-MULTIVIAJE",
            [
                _pedido(
                    "B03-NORTE-8",
                    coordenada=COORDENADA_NORTE,
                    unidades=8,
                ),
                _pedido(
                    "B03-ESTE-8",
                    coordenada=COORDENADA_ESTE,
                    unidades=8,
                ),
                _pedido(
                    "B03-CERCANA-8",
                    coordenada=COORDENADA_CERCANA,
                    unidades=8,
                ),
            ],
            seed=15_103,
        ),
    )

    ventanas = CasoBenchmarkClasico(
        caso_id="B04_VENTANAS",
        categoria="VENTANAS",
        descripcion=(
            "Tres pedidos con ventanas diferenciadas dentro del turno."
        ),
        instancia=_instancia(
            "BENCH-B04-VENTANAS",
            [
                _pedido(
                    "B04-NORTE-TEMPRANO",
                    coordenada=COORDENADA_NORTE,
                    unidades=3,
                    tiene_ventana_especifica=True,
                    hora_desde_min=465,
                    hora_hasta_min=510,
                ),
                _pedido(
                    "B04-ESTE-MEDIO",
                    coordenada=COORDENADA_ESTE,
                    unidades=3,
                    tiene_ventana_especifica=True,
                    hora_desde_min=525,
                    hora_hasta_min=585,
                ),
                _pedido(
                    "B04-CERCANA-TARDE",
                    coordenada=COORDENADA_CERCANA,
                    unidades=2,
                    tiene_ventana_especifica=True,
                    hora_desde_min=570,
                    hora_hasta_min=630,
                ),
            ],
            seed=15_104,
        ),
    )

    volcador = CasoBenchmarkClasico(
        caso_id="B05_VOLCADOR",
        categoria="VOLCADOR",
        descripcion=(
            "Combina pedidos normales y un pedido con volcador, que "
            "debe quedar último en su viaje."
        ),
        instancia=_instancia(
            "BENCH-B05-VOLCADOR",
            [
                _pedido(
                    "B05-NORMAL-NORTE",
                    coordenada=COORDENADA_NORTE,
                    unidades=2,
                ),
                _pedido(
                    "B05-VOLCADOR-ESTE",
                    coordenada=COORDENADA_ESTE,
                    unidades=6,
                    requiere_volcador=True,
                ),
                _pedido(
                    "B05-NORMAL-CERCANA",
                    coordenada=COORDENADA_CERCANA,
                    unidades=4,
                ),
            ],
            seed=15_105,
        ),
    )

    split = CasoBenchmarkClasico(
        caso_id="B06_SPLIT",
        categoria="SPLIT",
        descripcion=(
            "Pedido original dividido en dos partes, junto con otro "
            "pedido que permite distintas combinaciones de viajes."
        ),
        instancia=_instancia(
            "BENCH-B06-SPLIT",
            [
                _pedido(
                    "B06-SPLIT-P1",
                    pedido_original_id="B06-SPLIT-ORIGINAL",
                    numero_parte=1,
                    total_partes=2,
                    coordenada=COORDENADA_NORTE,
                    unidades=8,
                ),
                _pedido(
                    "B06-SPLIT-P2",
                    pedido_original_id="B06-SPLIT-ORIGINAL",
                    numero_parte=2,
                    total_partes=2,
                    coordenada=COORDENADA_NORTE,
                    unidades=4,
                ),
                _pedido(
                    "B06-ESTE-4",
                    coordenada=COORDENADA_ESTE,
                    unidades=4,
                ),
            ],
            seed=15_106,
        ),
    )

    return (
        simple,
        carga_paralela,
        multiviaje,
        ventanas,
        volcador,
        split,
    )
