"""Monitor: calcula sinais por dia, aplica limiares e so alerta apos `patience` dias seguidos.

Colunas esperadas: tokens_prompt, score_modelo (numericas), categoria (texto), latency_ms, erro (0/1),
feedback (1/0 ou NaN quando nao ha rotulo), custo.
Duas etapas separadas de proposito: `raw` guarda os numeros brutos do dia; `flags_from_raw` aplica os
limiares. Assim e possivel REPRODUZIR a decisao com outra configuracao sem recalcular as estatisticas.
"""
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from . import drift

NUMERIC = ("tokens_prompt", "score_modelo")
CATEGORICAL = ("categoria",)


@dataclass
class MonitorConfig:
    method: str = "both"            # psi | stat | both  (como decidir drift de entrada)
    psi_alert: float = 0.2
    alpha: float = 0.001            # teste estatistico de drift
    patience: int = 2               # dias seguidos com sinal antes de alertar
    latency_ratio: float = 1.25     # p95 do dia / p95 da referencia
    error_alpha: float = 0.001
    quality_alpha: float = 0.01
    cost_ratio: float = 1.25


@dataclass
class DayReport:
    day: int
    raw: dict
    flags: dict
    active_alerts: list[str] = field(default_factory=list)
    new_alerts: list[str] = field(default_factory=list)


class Reference:
    """Perfil da janela de referencia (por exemplo, a semana em que o modelo estava saudavel)."""

    def __init__(self, df: pd.DataFrame):
        if len(df) < 500:
            raise ValueError("Referencia pequena demais (minimo 500 linhas)")
        self.df = df.reset_index(drop=True)
        self.p95 = float(np.percentile(df["latency_ms"], 95))
        self.err = float(df["erro"].mean())
        fb = df["feedback"].dropna()
        self.quality = float(fb.mean()) if len(fb) else float("nan")
        self.cost = float(df["custo"].mean())


def day_stats(ref: Reference, cur: pd.DataFrame) -> dict:
    raw: dict = {}
    for f in NUMERIC:
        raw[f] = {"psi": drift.psi_numeric(ref.df[f], cur[f]), "p": drift.ks_pvalue(ref.df[f], cur[f])}
    for f in CATEGORICAL:
        raw[f] = {"psi": drift.psi_categorical(ref.df[f], cur[f]), "p": drift.chi2_pvalue(ref.df[f], cur[f])}
    raw["latencia_p95"] = {"ratio": float(np.percentile(cur["latency_ms"], 95)) / ref.p95}
    k, n = int(cur["erro"].sum()), len(cur)
    raw["taxa_erro"] = {"p": float(stats.binomtest(k, n, max(ref.err, 1e-4), alternative="greater").pvalue)}
    fb = cur["feedback"].dropna()
    if len(fb) >= 20 and not np.isnan(ref.quality):
        p = float(stats.binomtest(int(fb.sum()), len(fb), ref.quality, alternative="less").pvalue)
    else:
        p = 1.0
    raw["qualidade"] = {"p": p, "n": int(len(fb)), "taxa": float(fb.mean()) if len(fb) else float("nan")}
    raw["custo_medio"] = {"ratio": float(cur["custo"].mean()) / ref.cost}
    return raw


def flags_from_raw(raw: dict, cfg: MonitorConfig) -> dict[str, bool]:
    flags: dict[str, bool] = {}
    for f in NUMERIC + CATEGORICAL:
        a = raw[f]["psi"] >= cfg.psi_alert
        b = raw[f]["p"] < cfg.alpha
        flags[f] = {"psi": a, "stat": b, "both": a and b}[cfg.method]
    flags["latencia_p95"] = raw["latencia_p95"]["ratio"] > cfg.latency_ratio
    flags["taxa_erro"] = raw["taxa_erro"]["p"] < cfg.error_alpha
    flags["qualidade"] = raw["qualidade"]["p"] < cfg.quality_alpha
    flags["custo_medio"] = raw["custo_medio"]["ratio"] > cfg.cost_ratio
    return flags


def alerts_over_time(flags_by_day: list[dict[str, bool]], patience: int) -> list[list[str]]:
    """Para cada dia, a lista de sinais com `patience` ou mais dias seguidos sinalizados."""
    streak: dict[str, int] = {}
    out = []
    for flags in flags_by_day:
        for k, v in flags.items():
            streak[k] = streak.get(k, 0) + 1 if v else 0
        out.append(sorted(k for k, s in streak.items() if s >= patience))
    return out


class Monitor:
    def __init__(self, reference: pd.DataFrame, config: MonitorConfig | None = None):
        self.ref = Reference(reference)
        self.cfg = config or MonitorConfig()
        self.history: list[DayReport] = []
        self._streak: dict[str, int] = {}

    def observe(self, day_df: pd.DataFrame) -> DayReport:
        if len(day_df) < 50:
            raise ValueError("Poucas linhas no dia (minimo 50)")
        raw = day_stats(self.ref, day_df)
        flags = flags_from_raw(raw, self.cfg)
        for k, v in flags.items():
            self._streak[k] = self._streak.get(k, 0) + 1 if v else 0
        active = sorted(k for k, s in self._streak.items() if s >= self.cfg.patience)
        before = set(self.history[-1].active_alerts) if self.history else set()
        rep = DayReport(len(self.history), raw, flags, active, sorted(set(active) - before))
        self.history.append(rep)
        return rep
