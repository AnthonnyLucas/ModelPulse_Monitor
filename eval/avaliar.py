"""Experimento: falso alarme e atraso de deteccao de varias configuracoes, em fluxos simulados.

Para cada cenario e cada semente (30 por padrao), gera 7 dias de referencia + 30 dias de monitoramento,
calcula as estatisticas diarias UMA vez e aplica 4 configuracoes de decisao sobre os mesmos numeros.
  - detectou : o sinal-alvo ficou em alerta em algum dia a partir do inicio do problema
  - atraso   : mediana de (dia do primeiro alerta - dia de inicio), so entre os que detectaram
  - falso    : qualquer alerta (de qualquer sinal) ANTES do inicio do problema
"""
from __future__ import annotations
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "src"))

from modelmonitor.monitor import MonitorConfig, Reference, alerts_over_time, day_stats, flags_from_raw
from modelmonitor.simulate import CENARIOS, gerar_fluxo

CONFIGS = {
    "psi / 1 dia": MonitorConfig(method="psi", patience=1),
    "psi / 2 dias": MonitorConfig(method="psi", patience=2),
    "estat. / 2 dias": MonitorConfig(method="stat", patience=2),
    "ambos / 2 dias": MonitorConfig(method="both", patience=2),
}
INPUT_SIGNALS = {"tokens_prompt", "score_modelo", "categoria"}


def correr(n_seeds: int = 30, n_dias: int = 30, cenarios=None, batch: int = 500) -> dict:
    out: dict = {}
    cenarios = cenarios or [c for c in CENARIOS if c != "drift_irrelevante"]
    for cen in cenarios:
        inicio, alvo, _ = CENARIOS[cen]
        acc = {c: {"detectou": 0, "atrasos": [], "falso": 0, "dias_alerta_falso": 0, "entrada_detectou": 0}
               for c in CONFIGS}
        for seed in range(n_seeds):
            ref, dias = gerar_fluxo(cen, seed, n_dias, batch)
            r = Reference(ref)
            raws = [day_stats(r, d) for d in dias]
            for nome, cfg in CONFIGS.items():
                al = alerts_over_time([flags_from_raw(x, cfg) for x in raws], cfg.patience)
                pre = al[: inicio] if inicio is not None else al
                acc[nome]["falso"] += any(pre)
                acc[nome]["dias_alerta_falso"] += sum(bool(a) for a in pre)
                if inicio is None:
                    continue
                dia_alvo = next((d for d in range(inicio, n_dias) if alvo in al[d]), None)
                if dia_alvo is not None:
                    acc[nome]["detectou"] += 1
                    acc[nome]["atrasos"].append(dia_alvo - inicio)
                if any(set(al[d]) & INPUT_SIGNALS for d in range(inicio, n_dias)):
                    acc[nome]["entrada_detectou"] += 1
        out[cen] = {c: {
            "taxa_deteccao": round(a["detectou"] / n_seeds, 3) if inicio is not None else None,
            "atraso_mediano_dias": float(np.median(a["atrasos"])) if a["atrasos"] else None,
            "atraso_p90_dias": float(np.percentile(a["atrasos"], 90)) if a["atrasos"] else None,
            "atraso_max_dias": int(max(a["atrasos"])) if a["atrasos"] else None,
            "sementes_com_falso_alarme": a["falso"],
            "dias_falsos_por_semente": round(a["dias_alerta_falso"] / n_seeds, 3),
            "entrada_detectou": a["entrada_detectou"] / n_seeds if inicio is not None else None,
        } for c, a in acc.items()}
        out[cen]["_meta"] = {"inicio": inicio, "alvo": alvo, "n_seeds": n_seeds}
    return out


def imprimir(res: dict) -> None:
    n = res["controle"]["_meta"]["n_seeds"]
    print(f"\n== A) FALSO ALARME no controle (sem problema), {n} sementes x 30 dias")
    print(f"{'config':<18}{'sementes c/ alarme':>20}{'dias falsos/semente':>22}")
    for c, v in res["controle"].items():
        if c != "_meta":
            print(f"{c:<18}{v['sementes_com_falso_alarme']:>14}/{n:<5}{v['dias_falsos_por_semente']:>22.3f}")
    print(f"\n== B) DETECCAO por cenario: taxa de deteccao / atraso mediano (p90) em dias")
    cols = list(CONFIGS)
    print(f"{'cenario (alvo)':<34}" + "".join(f"{c:>22}" for c in cols))
    for cen, v in res.items():
        if cen == "controle":
            continue
        m = v["_meta"]
        cells = []
        for c in cols:
            d = v[c]
            at = "-" if d["atraso_mediano_dias"] is None else f"{d['atraso_mediano_dias']:.0f}d (p90 {d['atraso_p90_dias']:.0f})"
            cells.append(f"{d['taxa_deteccao']:.2f} / {at}")
        print(f"{cen + ' (' + m['alvo'] + ')':<34}" + "".join(f"{x:>22}" for x in cells))
    q = res["queda_qualidade"]["ambos / 2 dias"]
    print(f"\n== C) PONTO CEGO: queda de qualidade com entradas inalteradas (config 'ambos / 2 dias')")
    print(f"   monitor de qualidade detectou: {q['taxa_deteccao']:.2f}   |   monitores de drift de entrada detectaram: {q['entrada_detectou']:.2f}")


def imprimir_trafego_alto(res: dict) -> None:
    v = res["drift_irrelevante"]
    n = v["_meta"]["n_seeds"]
    print(f"\n== D) TRAFEGO ALTO (5000 req/dia): mudanca de apenas +3% nos tokens, sem efeito pratico ({n} sementes)")
    print(f"{'config':<18}{'alertou (taxa)':>16}{'atraso mediano':>16}")
    for c in CONFIGS:
        d = v[c]
        at = "-" if d["atraso_mediano_dias"] is None else f"{d['atraso_mediano_dias']:.0f}d"
        print(f"{c:<18}{d['taxa_deteccao']:>16.2f}{at:>16}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    res = correr(n)
    (BASE / "results").mkdir(exist_ok=True)
    (BASE / "results" / "experimento.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    imprimir(res)
    alto = correr(max(5, n // 3), n_dias=15, cenarios=["drift_irrelevante"], batch=5000)
    (BASE / "results" / "trafego_alto.json").write_text(json.dumps(alto, indent=2), encoding="utf-8")
    imprimir_trafego_alto(alto)
