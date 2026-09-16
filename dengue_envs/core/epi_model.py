"""Curvas epidêmicas temporais: SEIR com parâmetros de literatura.

Por que este módulo existe
--------------------------
Os dois geradores (`World` sintético e `KrigingWorld`) usavam um SIR com dois
erros que se compensavam num pico falso:

1. ``-beta * S * I`` **sem dividir por N**. `N` era recebido e nunca usado. Com
   ``beta = R0 * gamma``, o número de reprodução efetivo vira ``R0 * N`` — com
   N = 150, um R0 declarado de 1,5 valia **225** na prática (1,2 → 180 na chik).
2. ``gamma = 0.004``, isto é, um período infeccioso de **250 dias**. Viremia de
   dengue dura ~4–5 dias.

Resultado medido: a epidemia notificava todos os ~140 casos em **9 dias**, com
pico no dia 3, apesar de ``epilength: 60``. Toda a dinâmica temporal do ambiente
— sobreposição de casos, tempo até o laudo, fase da epidemia — estava calibrada
a esse pico artificial. Corrigir só o ``/N`` não basta: com 250 dias de período
infeccioso a epidemia nem decola (3 casos em 60 dias).

O modelo
--------
SEIR em população fechada, com transmissão dependente de frequência::

    S' = -beta S I / N
    E' =  beta S I / N - sigma E
    I' =  sigma E - gamma I
    R' =  gamma I

A notificação acompanha ``sigma E`` (início da fase infecciosa ≈ início dos
sintomas). ``beta = R0 * gamma``, que só é o R0 verdadeiro com o ``/ N``.

**Vetor implícito.** Dengue e chikungunya são transmitidas por *Aedes*, e o
intervalo entre casos sucessivos inclui o ciclo no mosquito (período de
incubação extrínseco). Em vez de um modelo hospedeiro-vetor completo, o período
latente humano aqui é **efetivo**: ele absorve a incubação intrínseca e a
extrínseca, de modo que o tempo de geração (``latent + infectious``) bata com o
intervalo serial observado. Isso preserva o *tempo* da epidemia, que é o que o
ambiente precisa, com dois parâmetros por doença em vez de ~8.

Parâmetros e fontes (todas verificadas no texto original)
--------------------------------------------------------
Dengue
  - incubação intrínseca: "The mean IIP estimate was 5.9 days, with 95%
    expected between days 3 and 10."
    Chan M, Johansson MA. PLoS One. 2012;7(11):e50972.
    doi:10.1371/journal.pone.0050972
  - infecciosidade humana: "humans can be infectious to mosquitoes from 1.5
    days prior to the onset of symptoms to around 5 days after the
    commencement of symptoms".
    Carrington LB, Simmons CP. Front Immunol. 2014;5:290.
    doi:10.3389/fimmu.2014.00290
  - intervalo serial: "The strongest spatial clustering occurred at the 15–17
    day interval."
    Aldstadt J et al. Trop Med Int Health. 2012;17(9):1076–1085.
    doi:10.1111/j.1365-3156.2012.03040.x
  - R0 no Rio de Janeiro: 1,70 (IC95% 1,50–2,02) em 2002 e 1,25 (1,18–1,36)
    em 2012.
    Villela DAM, Bastos LS, de Carvalho LM, Cruz OG, Gomes MFC, Durovni B,
    Lemos MC, Saraceni V, Coelho FC, Codeço CT. Epidemiol Infect.
    2017;145(8):1649–1657. doi:10.1017/S0950268817000358
  => ``infectious_days = 5``, ``latent_days = 11`` (tempo de geração 16 dias,
     centro do intervalo serial de Aldstadt). R0 em [1,25; 1,70].

Chikungunya
  - R0 no Rio de Janeiro: "Assuming a generation time (GT) of 14 days, we
    estimated a R0 of 1.56 (95% CI = 1.46–1.67)."
    Moreira FRR, de Menezes MT, Salgado-Benvindo C, et al. PLoS Negl Trop
    Dis. 2023;17(9):e0011536. doi:10.1371/journal.pntd.0011536
  => tempo de geração 14 dias, R0 em [1,46; 1,67].

  *Ressalvas:* (1) os 14 dias são uma SUPOSIÇÃO do próprio artigo, não uma
  medição de intervalo serial. (2) a divisão entre latente e infeccioso
  (8 + 6) não tem fonte direta verificada; só a soma está ancorada.

Nota sobre o cenário: pela literatura, o R0 da chik no Rio (1,56) é comparável
ao da dengue (1,25–1,70), não sistematicamente menor. O ambiente original
tratava a chik como surto secundário sempre menor — isso era escolha de
desenho, não achado epidemiológico.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.integrate import odeint


@dataclass(frozen=True)
class DiseaseParams:
    """Parâmetros temporais de uma doença (em dias)."""

    latent_days: float
    infectious_days: float
    r0_range: Tuple[float, float]

    @property
    def generation_time(self) -> float:
        """Tempo de geração médio do SEIR com estágios exponenciais."""
        return self.latent_days + self.infectious_days


DENGUE = DiseaseParams(latent_days=11.0, infectious_days=5.0, r0_range=(1.25, 1.70))
CHIK = DiseaseParams(latent_days=8.0, infectious_days=6.0, r0_range=(1.46, 1.67))


def final_size(r0: float) -> float:
    """Fração final atingida numa população inteiramente suscetível.

    Resolve ``z = 1 - exp(-R0 z)`` (vale para SIR e SEIR fechados). Útil para
    conferir que a curva integrada termina no tamanho que a teoria prevê.
    """
    if r0 <= 1.0:
        return 0.0
    z = 0.5
    for _ in range(200):
        z = 1.0 - np.exp(-r0 * z)
    return float(z)


def seir_cumulative_cases(
    r0: float,
    params: DiseaseParams,
    population: float,
    epilength: int,
    initial_infected_fraction: float = 0.01,
) -> np.ndarray:
    """Casos notificados acumulados por dia, no formato que os geradores esperam.

    Devolve um vetor de comprimento ``epilength`` em **unidades de casos**
    (não fração), crescente, começando em 0 — exatamente o que
    `build_case_series` consome ao arredondar e tirar diferenças diárias.

    ``initial_infected_fraction`` representa a circulação já em curso no início
    da temporada. Dengue é endêmica no Rio: uma temporada não começa de um caso
    importado isolado. Semear com fração pequena encurta a fase de latência
    inicial sem alterar o R0 nem o tempo de geração.
    """
    if population <= 0:
        raise ValueError(f"population deve ser > 0; recebido {population}")
    if not 0.0 < initial_infected_fraction < 1.0:
        raise ValueError(
            "initial_infected_fraction deve estar em (0, 1); "
            f"recebido {initial_infected_fraction}"
        )
    sigma = 1.0 / params.latent_days
    gamma = 1.0 / params.infectious_days
    beta = float(r0) * gamma
    n = float(population)

    # Distribui a infecção inicial entre E e I na proporção das durações,
    # que é aproximadamente o equilíbrio da fase de crescimento.
    i0 = n * initial_infected_fraction
    frac_e = params.latent_days / params.generation_time
    e0, i0 = i0 * frac_e, i0 * (1.0 - frac_e)
    s0 = n - e0 - i0

    def deriv(y, _t):
        s, e, i, _r, _c = y
        forca = beta * s * i / n
        return [-forca, forca - sigma * e, sigma * e - gamma * i, gamma * i, sigma * e]

    y = odeint(deriv, [s0, e0, i0, 0.0, 0.0], np.arange(0, int(epilength)))
    return np.maximum.accumulate(y[:, 4])


def legacy_sir_cumulative(popsize: int, epilength: int, r0: float, i0: int = 10) -> np.ndarray:
    """O SIR original, **com os dois erros**, preservado só para reprodução.

    Existe para que resultados anteriores continuem reproduzíveis com
    ``epi_model: legacy``. Não usar em experimento novo: o R0 efetivo é
    ``R0 * popsize`` e o período infeccioso é de 250 dias.
    """

    def sir(y, _t, beta, gamma, _n):
        s, i, _inc, _r = y
        return [-beta * s * i, beta * s * i - gamma * i, beta * s * i, gamma * i]

    gamma = 0.004
    beta = float(r0) * gamma
    y = odeint(
        sir,
        [popsize - i0, i0, 0, 0],
        np.arange(0, epilength),
        args=(beta, gamma, popsize),
    )
    return y[:, 2]
