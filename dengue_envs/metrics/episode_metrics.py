"""Métricas de acurácia por passo e métricas clínicas/econômicas de episódio.

Extração fiel de `DengueDiagnosticsEnv.calc_accuracy` e `get_episode_metrics`
como funções puras, operando sobre listas/arrays. Facilita reuso na avaliação
(benchmark) e testes, sem depender de uma instância do ambiente.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def step_accuracy(true: List[dict], estimated: List[tuple]) -> Tuple[float, float]:
    """Acurácia média (dengue/chik) e MAPE de contagem para um passo.

    - `true`: lista de dicts com chave 'disease'.
    - `estimated`: lista de tuplas onde o índice 2 é o diagnóstico estimado.
    Retorna (mean_accuracy, mape).
    """
    if len(true) == 0:
        return 0.0, 0.0

    tpd = fpd = tnd = fnd = 0
    tpc = fpc = tnc = fnc = 0

    for t, e in zip(true, estimated):
        true_label = t["disease"]
        est_label = e[2]

        is_dengue = true_label == 0
        predicted_dengue = est_label == 0
        if is_dengue and predicted_dengue:
            tpd += 1
        elif is_dengue and not predicted_dengue:
            fnd += 1
        elif not is_dengue and predicted_dengue:
            fpd += 1
        else:
            tnd += 1

        is_chik = true_label == 1
        predicted_chik = est_label == 1
        if is_chik and predicted_chik:
            tpc += 1
        elif is_chik and not predicted_chik:
            fnc += 1
        elif not is_chik and predicted_chik:
            fpc += 1
        else:
            tnc += 1

    total_cases = len(true)
    accuracy_dengue = (tpd + tnd) / total_cases
    accuracy_chik = (tpc + tnc) / total_cases
    mean_accuracy = (accuracy_dengue + accuracy_chik) / 2

    true_numdengue = len([c for c in true if c["disease"] == 0])
    estimated_numdengue = len([c for c in estimated if c[2] == 0])
    true_chik = len([c for c in true if c["disease"] == 1])
    estimated_chik = len([c for c in estimated if c[2] == 1])
    true_total = true_numdengue + true_chik
    est_total = estimated_numdengue + estimated_chik

    if true_total > 0:
        mape = np.abs(true_total - est_total) / true_total
    else:
        mape = 0.0

    return mean_accuracy, mape


def episode_metrics(y_true, y_pred, total_tests: int, total_reward: float) -> Dict:
    """Métricas clínicas e econômicas ao fim do episódio.

    Espelha `get_episode_metrics`, tratando dengue (classe 0) como "positivo".
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    TP = np.sum((y_true == 0) & (y_pred == 0))
    TN = np.sum((y_true != 0) & (y_pred != 0))
    FP = np.sum((y_true != 0) & (y_pred == 0))
    FN = np.sum((y_true == 0) & (y_pred != 0))

    epsilon = 1e-7

    sensitivity = TP / (TP + FN + epsilon)
    specificity = TN / (TN + FP + epsilon)
    precision = TP / (TP + FP + epsilon)
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity + epsilon)
    accuracy = (TP + TN) / len(y_true) if len(y_true) > 0 else 0.0

    total_cases = len(y_true)
    test_cost = total_tests * 1.0
    total_correct = TP + TN
    cost_per_correct_diagnosis = test_cost / (total_correct + epsilon)
    potential_tests = total_cases * 2
    test_reduction_rate = 1 - (total_tests / potential_tests) if potential_tests > 0 else 0.0

    return {
        "Acurácia": accuracy,
        "Sensibilidade (Dengue)": sensitivity,
        "Especificidade": specificity,
        "F1-Score": f1_score,
        "Precisão": precision,
        "Custo Total de Testes": test_cost,
        "Testes Realizados": total_tests,
        "Custo por Acerto": cost_per_correct_diagnosis,
        "Redução de Testes (%)": test_reduction_rate * 100,
        "Recompensa Total": total_reward,
    }
