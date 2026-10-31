"""Wrapper de compatibilidad que delega el entrenamiento en `scripts.rl.train_rl`."""

from __future__ import annotations

from scripts.rl.train_rl import (
    CONFIGS_DIR,
    DEFAULT_CONFIG_PATH,
    HISTORIAL_DIR,
    MAX_PEDIDOS_V2,
    MODELS_DIR,
    EntornoCurricularMultiInstancia,
    cargar_configuracion_entrenamiento,
    construir_instancia_entrenamiento,
    ejecutar_entrenamiento_desde_config,
    ejecutar_entrenamiento_v2,
    evaluar_lote_comparativo,
    fijar_semillas_globales,
    main,
    respaldar_modelo_v1,
    transferir_pesos_v1_a_v2,
)

__all__ = [
    "CONFIGS_DIR",
    "DEFAULT_CONFIG_PATH",
    "HISTORIAL_DIR",
    "MAX_PEDIDOS_V2",
    "MODELS_DIR",
    "EntornoCurricularMultiInstancia",
    "cargar_configuracion_entrenamiento",
    "construir_instancia_entrenamiento",
    "ejecutar_entrenamiento_desde_config",
    "ejecutar_entrenamiento_v2",
    "evaluar_lote_comparativo",
    "fijar_semillas_globales",
    "main",
    "respaldar_modelo_v1",
    "transferir_pesos_v1_a_v2",
]

if __name__ == "__main__":
    raise SystemExit(main())
