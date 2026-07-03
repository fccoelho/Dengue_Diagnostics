"""Núcleo reutilizável do ambiente: clínica, confirmação epi, recompensa e casos.

Módulos NOVOS (Fase 1) que espelham a lógica já validada do
`DengueDiagnosticsEnv`, isolados para teste e reuso. O ambiente legado continua
funcionando sem depender deste pacote; a ligação será feita numa fase posterior.
"""
from dengue_envs.core.case_store import OBS_COLUMNS, sync_obs_cases
from dengue_envs.core.clinical import ClinicalModel, update_case_status
from dengue_envs.core.epi_confirm import epi_confirm
from dengue_envs.core.reward import DEFAULT_COSTS, RewardEngine

__all__ = [
    "ClinicalModel",
    "update_case_status",
    "epi_confirm",
    "RewardEngine",
    "DEFAULT_COSTS",
    "sync_obs_cases",
    "OBS_COLUMNS",
]
