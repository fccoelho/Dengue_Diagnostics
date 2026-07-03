# Wrappers legados (templates baseados no pacote antigo `gym`).
# Mantidos intactos, mas importados de forma tolerante: se o `gym` antigo não
# estiver instalado, o pacote continua importável e os wrappers novos (Fase 1)
# permanecem disponíveis.
try:  # pragma: no cover - depende do pacote antigo `gym`
    from dengue_envs.wrappers.clip_reward import ClipReward
    from dengue_envs.wrappers.discrete_actions import DiscreteActions
    from dengue_envs.wrappers.reacher_weighted_reward import ReacherRewardWrapper
    from dengue_envs.wrappers.relative_position import RelativePosition
except Exception:  # ImportError e afins (gym ausente)
    ClipReward = None
    DiscreteActions = None
    ReacherRewardWrapper = None
    RelativePosition = None

# Wrappers canônicos da Fase 1 (dependem apenas de gymnasium).
from dengue_envs.wrappers.case_by_case import CaseByCaseWrapper
from dengue_envs.wrappers.factory import make_env, make_raw_env
from dengue_envs.wrappers.map_tensor import DengueWrapper

__all__ = [
    "DengueWrapper",
    "CaseByCaseWrapper",
    "make_env",
    "make_raw_env",
]
