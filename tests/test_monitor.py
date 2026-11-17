import numpy as np
import pytest
from modelmonitor import drift
from modelmonitor.monitor import (Monitor, MonitorConfig, Reference, alerts_over_time, day_stats, flags_from_raw)
from modelmonitor.simulate import CENARIOS, Estado, gerar_dia, gerar_fluxo


def rodar(cenario, seed=0, cfg=None, n_dias=30):
    ref, dias = gerar_fluxo(cenario, seed, n_dias)
    m = Monitor(ref, cfg)
    return [m.observe(d) for d in dias]


# ---------- estatisticas ----------
def test_psi_numerico_estavel_e_deslocado():
    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 1, 5000), rng.normal(0, 1, 5000)
    assert drift.psi_numeric(a, b) < 0.05
    assert drift.psi_numeric(a, rng.normal(1.0, 1, 5000)) > 0.25


def test_psi_categorico():
    ref = ["a"] * 50 + ["b"] * 50
    assert drift.psi_categorical(ref, ref) == pytest.approx(0, abs=1e-9)
    assert drift.psi_categorical(ref, ["a"] * 90 + ["b"] * 10) > 0.25
    assert drift.psi_categorical(ref, ["a"] * 50 + ["c"] * 50) > 0.25      # categoria nova nao quebra


def test_ks_e_chi2():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 2000)
    assert drift.ks_pvalue(a, rng.normal(0.5, 1, 500)) < 1e-6
    assert drift.chi2_pvalue(["a"] * 100 + ["b"] * 100, ["a"] * 180 + ["b"] * 20) < 1e-6


def test_referencia_pequena_e_dia_pequeno():
    ref, dias = gerar_fluxo("controle", 0, 2)
    with pytest.raises(ValueError):
        Reference(ref.head(100))
    with pytest.raises(ValueError):
        Monitor(ref).observe(dias[0].head(10))


# ---------- monitor ----------
def test_controle_nao_gera_alerta_seed0():
    assert all(not r.active_alerts for r in rodar("controle"))


@pytest.mark.parametrize("cenario", ["drift_entrada", "drift_categorias", "queda_qualidade", "latencia", "aumento_erros"])
def test_problemas_injetados_sao_detectados_logo(cenario):
    inicio, alvo, _ = CENARIOS[cenario]
    reps = rodar(cenario)
    assert all(not r.active_alerts for r in reps[:inicio])               # nada antes do problema
    primeiro = next(d for d in range(inicio, 30) if alvo in reps[d].active_alerts)
    # Qualidade usa so ~100 feedbacks/dia, entao o atraso varia mais entre sementes (seed 0 leva 6 dias).
    limite = 10 if alvo == "qualidade" else 3
    assert primeiro - inicio <= limite


def test_queda_de_qualidade_e_ponto_cego_dos_monitores_de_entrada():
    inicio, _, _ = CENARIOS["queda_qualidade"]
    reps = rodar("queda_qualidade")
    entrada = {"tokens_prompt", "score_modelo", "categoria"}
    assert not any(set(r.active_alerts) & entrada for r in reps)
    assert any("qualidade" in r.active_alerts for r in reps[inicio:])


def test_paciencia_evita_alerta_de_pico_de_um_dia():
    ref, dias = gerar_fluxo("controle", 3, 12)
    rng = np.random.default_rng(99)
    dias[6] = gerar_dia(rng, 500, Estado(tokens_mult=1.5))                # um unico dia ruim
    for pac, esperado in ((1, True), (2, False)):
        m = Monitor(ref, MonitorConfig(patience=pac))
        alertou = any("tokens_prompt" in m.observe(d).active_alerts for d in dias)
        assert alertou is esperado


def test_new_alerts_so_aparece_na_primeira_vez():
    reps = rodar("latencia")
    novos = [r.day for r in reps if "latencia_p95" in r.new_alerts]
    assert len(novos) == 1


def test_alerts_over_time():
    f = [{"x": True}, {"x": True}, {"x": False}, {"x": True}]
    assert alerts_over_time(f, 2) == [[], ["x"], [], []]


def test_mais_paciencia_nunca_alerta_antes():
    ref, dias = gerar_fluxo("drift_entrada", 1, 30)
    raws = [day_stats(Reference(ref), d) for d in dias[:20]]
    cfg1, cfg2 = MonitorConfig(patience=1), MonitorConfig(patience=2)
    a1 = alerts_over_time([flags_from_raw(r, cfg1) for r in raws], 1)
    a2 = alerts_over_time([flags_from_raw(r, cfg2) for r in raws], 2)
    for x, y in zip(a1, a2):
        assert set(y) <= set(x)


# ---------- experimento ----------
def test_experimento_reproduz_tradeoff_medido():
    """Trava o achado: 'ambos' (PSI e teste) perde o drift sutil que o teste estatistico pega.
    Se isso mudar, revise o README."""
    from eval.avaliar import correr
    r = correr(n_seeds=5, n_dias=20, cenarios=["controle", "drift_entrada", "drift_sutil"])
    assert r["drift_entrada"]["ambos / 2 dias"]["taxa_deteccao"] == 1.0
    assert r["drift_sutil"]["estat. / 2 dias"]["taxa_deteccao"] == 1.0
    assert r["drift_sutil"]["ambos / 2 dias"]["taxa_deteccao"] == 0.0
    assert r["controle"]["ambos / 2 dias"]["sementes_com_falso_alarme"] == 0
