"""Baseline "confirmar tudo" — a política fixa mais forte sob a recompensa v2.

Conclui todos os casos com o palpite clínico corrente, sem pedir exames: se o
médico disse dengue, a ação é "concluir dengue"; se disse chik, "concluir chik";
etc. (ação = 4 + ``agent_diagnosis``). Não melhora a acurácia (ela continua
sendo a do médico), mas coleta o bônus de decisão correta (+10) sempre que a
triagem clínica acerta.

É o baseline mais duro de bater e, por isso, o que realmente mede se o agente
aprendeu algo: sob a recompensa v2 ele rende ~+129, contra ~-142 do `testall` e
~-95 do `clinical`. Um agente de RL que não supere este número não está
agregando valor — está apenas replicando "aceite o médico e assine embaixo".

Contraponto importante: é uma política de alta variância. Com um médico bom
rende ~+1400; com um médico ruim, ~-2300. Superá-lo exige justamente o que o
problema tem de difícil — inferir a competência do médico ao longo do episódio
e testar apenas quando ela for baixa.

Nota (v3, casos não-arbovirose): esta política nunca conclui "outro" por conta
própria, mesmo quando o palpite clínico já é ``OTHER`` (o médico raramente
reconhece isso sozinho — ver ``other_recognition_prob``). Ela mede o piso da
triagem clínica pura, não uma estratégia de vigilância completa.
"""
from __future__ import annotations

from agents.base import EpisodeRunner

# Ações conclusivas no CaseByCaseWrapper: 4=dengue, 5=chik, 6=outro.
CONCLUDE_DENGUE = 4


class ConfirmAllAgentRunner(EpisodeRunner):
    """Baseline que conclui todos os casos com o palpite clínico corrente."""

    name = "confirmall"

    def choose_action(self, env) -> int:
        base = env.unwrapped
        case_id = env.current_case[0]
        if case_id not in base.obs_cases.index:
            return CONCLUDE_DENGUE
        diagnosis = int(base.obs_cases.loc[case_id, "agent_diagnosis"])
        return CONCLUDE_DENGUE + diagnosis
