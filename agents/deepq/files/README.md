# Agente DQN para Diagnóstico de Dengue

Esta seção contém a implementação de um agente de Deep Q-Network (DQN) usando a biblioteca Tianshou, projetado para aprender a política ótima de testagem no ambiente customizado `DengueDiagnosticsEnv`.

## Descrição do Problema

O ambiente `DengueDiagnosticsEnv` é complexo:

1.  **Observação Espacial:** O estado é um mapa 2D (`400x400`) que evolui ao longo do tempo.
2.  **Múltiplas Ações:** Em cada passo de tempo (`t`), múltiplos casos de arbovirose podem aparecer. O agente precisa decidir uma ação para *cada um* desses casos.
3.  **Espaço de Ação:** O espaço de ação original é um `spaces.Sequence`, esperando uma lista (ou tupla) de decisões no formato `(case_id, action_id)`.

Isso não é compatível com a `DQNPolicy` padrão do Tianshou, que espera um espaço de ação `Discrete` simples.

## Arquitetura da Solução

Para resolver esse desafio, o problema foi reformulado usando uma cadeia de *wrappers* do Gymnasium e uma rede neural customizada:

### 1. `dengue_wrappers.py`

Este arquivo é a chave da solução e contém dois wrappers:

* **`DengueWrapper(gym.ObservationWrapper)`**:
    * Converte a observação em dicionário (`Dict`) do ambiente base.
    * Transforma o estado em um único tensor (mapa) de 4 canais com shape `(4, 400, 400)`, representando o diagnóstico clínico, status de testes e casos ativos.

* **`CaseByCaseWrapper(gym.Wrapper)`**:
    * **Reformulação do Problema**: Este wrapper "serializa" o processo de decisão.
    * Ele intercepta o `step` e, em vez de pedir *N* decisões ao agente de uma vez, ele itera sobre cada caso ativo *um por um*.
    * **Novo Espaço de Ação**: Para o agente, o espaço de ação se torna um `spaces.Discrete(6)` (Testar Dengue, Testar Chik, etc.).
    * **Nova Observação**: A observação do agente se torna um `Dict` contendo o mapa `(4, 400, 400)` e as coordenadas `(x, y)` do caso específico que está sendo decidido no momento.
    * Ele coleta todas as decisões individuais (ação 0-5) e as envia ao ambiente base na tupla `((case_id_1, action_1), ...)` que o ambiente espera.

### 2. `dengue_network.py`

* **`DengueNet(nn.Module)`**: É uma rede customizada que aceita a observação `Dict` do `CaseByCaseWrapper`.
* Possui dois "braços":
    1.  Um **Encoder CNN** que processa o mapa `(4, 400, 400)`.
    2.  Um **Encoder MLP** que processa as coordenadas `(2,)` do caso atual.
* Os *features* são concatenados e passados por um *head* MLP que retorna **6 Q-values**, um para cada ação possível para *aquele caso específico*.

### 3. `train_dqn.py`

* Script principal que conecta tudo.
* Usa o `OffpolicyTrainer`, `Collector`, e `DQNPolicy` padrões do Tianshou.
* Configura o `VectorReplayBuffer` para armazenar as transições (observações `Dict`).
---

## Como Executar

### 1. Instalação

Este projeto usa Poetry para gerenciamento de dependências.

```bash
# 1. Instale as dependências listadas no pyproject.toml
poetry install

# 2. Ative o ambiente virtual
poetry shell

# Inicia o script de treinamento
python train_dqn.py