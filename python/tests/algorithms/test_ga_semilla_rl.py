from __future__ import annotations
from planner.algorithms.ga import ConfiguracionGA, GeneticAlgorithmPlanner
from planner.core.schema import InstanciaTurno, PedidoInput, Turno
from planner.domain.validator import validar_plan

def _instancia() -> InstanciaTurno:
    pedidos = [PedidoInput(pedido_id=f'P{indice:03d}', pedido_original_id=f'P{indice:03d}', numero_parte=1, total_partes=1, turno=Turno.MANANA, latitud=-32.84 - indice * 0.001, longitud=-60.71 - indice * 0.001, unidades_capacidad=2, requiere_volcador=False, tiene_ventana_especifica=False, hora_desde_min=450, hora_hasta_min=720) for indice in range(1, 5)]
    return InstanciaTurno(instancia_id='GA-SEMILLA-RL', fecha_operacion='2026-08-27', turno=Turno.MANANA, pedidos=pedidos, lat_corralon=-32.8495006, lon_corralon=-60.722653, capacidad_camion=8, cantidad_camiones=2, hora_inicio_turno_min=450, hora_fin_objetivo_min=720, hora_fin_tolerancia_min=735, seed_escenario=7001, seed_ejecucion=1007001)

def test_ga_acepta_semilla_rl_y_puede_excluir_greedy():
    instancia = _instancia()
    semilla = tuple(reversed([pedido.pedido_id for pedido in instancia.pedidos]))
    planner = GeneticAlgorithmPlanner(configuracion_ga=ConfiguracionGA(tamano_poblacion=12, generaciones=8, tamano_elite=2, tamano_torneo=2, generaciones_sin_mejora_max=4), seed=16002, semillas_iniciales=(semilla,), incluir_semilla_greedy=False, fraccion_variantes_semilla=0.5)
    plan = planner.generar_plan(instancia)
    assert planner.semillas_iniciales_utilizadas == (semilla,)
    assert validar_plan(instancia, plan).valido
