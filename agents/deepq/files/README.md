# `agents/deepq/files` (legado / shims)

O path ativo do DQN é o pacote pai:

- `agents/deepq/network.py`
- `agents/deepq/agent.py`
- `agents/deepq/train.py`
- `agents/deepq/watch.py`

Aqui restam apenas:

- `fcn_network.py` — shim depreciado → `agents.deepq.network`
- `dengue_wrapper.py` — shim depreciado → `dengue_envs.wrappers`
- `*_old.py` — backups locais
- `logs/` — artefatos antigos de treino

Scripts de treino/watch antigos foram removidos do working tree; continuam
recuperáveis pelo histórico do git.
