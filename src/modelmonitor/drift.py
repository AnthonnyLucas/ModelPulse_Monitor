"""Estatisticas de drift entre uma janela de referencia e uma janela atual."""
from __future__ import annotations
import numpy as np
from scipy import stats

_EPS = 1e-4


def psi_numeric(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index com faixas por quantis da referencia.
    Regra de bolso: < 0,1 estavel; 0,1 a 0,25 mudanca moderada; > 0,25 mudanca grande."""
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, _EPS, None), np.clip(c, _EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_categorical(ref, cur) -> float:
    cats = sorted(set(ref) | set(cur))
    r = np.array([np.mean(np.asarray(ref) == k) for k in cats])
    c = np.array([np.mean(np.asarray(cur) == k) for k in cats])
    r, c = np.clip(r, _EPS, None), np.clip(c, _EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def ks_pvalue(ref, cur) -> float:
    """Teste de Kolmogorov-Smirnov de duas amostras (variaveis numericas)."""
    return float(stats.ks_2samp(ref, cur).pvalue)


def chi2_pvalue(ref, cur) -> float:
    """Qui-quadrado de homogeneidade (variaveis categoricas)."""
    cats = sorted(set(ref) | set(cur))
    table = np.array([[np.sum(np.asarray(x) == k) for k in cats] for x in (ref, cur)])
    table = table[:, table.sum(axis=0) > 0]
    return float(stats.chi2_contingency(table)[1])
