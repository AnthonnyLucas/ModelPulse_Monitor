"""Fluxo SINTETICO de um servico de IA, com problemas injetados em dia conhecido.

Tudo aqui e simulado: serve para medir os detectores (falso alarme e atraso), nao para dizer
nada sobre um sistema real. Suposicao importante: no cenario 'queda_qualidade' a distribuicao das
entradas e do score do modelo NAO muda (o modelo continua 'confiante' e piora), o que e um caso
classico de concept drift que monitores de entrada nao enxergam.
"""
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CATEGORIAS = ["pagamento", "entrega", "conta", "bug", "duvida"]
P_BASE = [0.25, 0.30, 0.15, 0.10, 0.20]


@dataclass
class Estado:
    tokens_mult: float = 1.0
    cat_p: list[float] = field(default_factory=lambda: list(P_BASE))
    qualidade: float = 0.82
    latencia_mult: float = 1.0
    erro: float = 0.01


def gerar_dia(rng: np.random.Generator, n: int, est: Estado, feedback_rate: float = 0.2) -> pd.DataFrame:
    tokens = rng.lognormal(np.log(220), 0.5, n) * est.tokens_mult
    latency = (300 + 1.8 * tokens) * rng.lognormal(0, 0.25, n) * est.latencia_mult
    observado = rng.random(n) < feedback_rate
    feedback = np.where(observado, (rng.random(n) < est.qualidade).astype(float), np.nan)
    return pd.DataFrame({
        "tokens_prompt": tokens,
        "score_modelo": rng.beta(8, 2, n),
        "categoria": rng.choice(CATEGORIAS, n, p=est.cat_p),
        "latency_ms": latency,
        "erro": (rng.random(n) < est.erro).astype(int),
        "feedback": feedback,
        "custo": tokens * 6e-6 + 2e-4,
    })


# nome -> (dia de inicio, sinal-alvo, funcao que altera o Estado)
def _tokens(e): e.tokens_mult = 1.4
def _tokens_sutil(e): e.tokens_mult = 1.15
def _tokens_irrelevante(e): e.tokens_mult = 1.03
def _categorias(e): e.cat_p = [0.15, 0.20, 0.10, 0.35, 0.20]
def _qualidade(e): e.qualidade = 0.70
def _latencia(e): e.latencia_mult = 1.5
def _erro(e): e.erro = 0.05


CENARIOS = {
    "controle": (None, None, None),
    "drift_entrada": (10, "tokens_prompt", _tokens),
    "drift_sutil": (10, "tokens_prompt", _tokens_sutil),
    "drift_irrelevante": (10, "tokens_prompt", _tokens_irrelevante),
    "drift_categorias": (12, "categoria", _categorias),
    "queda_qualidade": (14, "qualidade", _qualidade),
    "latencia": (16, "latencia_p95", _latencia),
    "aumento_erros": (18, "taxa_erro", _erro),
}


def gerar_fluxo(cenario: str, seed: int, n_dias: int = 30, batch: int = 500, ref_dias: int = 7):
    """Retorna (referencia, [dataframe de cada dia])."""
    rng = np.random.default_rng(seed)
    ref = pd.concat([gerar_dia(rng, batch, Estado()) for _ in range(ref_dias)], ignore_index=True)
    inicio, _, efeito = CENARIOS[cenario]
    dias = []
    for d in range(n_dias):
        est = Estado()
        if efeito and d >= inicio:
            efeito(est)
        dias.append(gerar_dia(rng, batch, est))
    return ref, dias
