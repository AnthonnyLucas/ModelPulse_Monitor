# Model Monitor: monitoramento de modelo em produção

Biblioteca em Python que vigia um serviço de IA no dia a dia e avisa quando algo muda: **drift das entradas**, **queda de qualidade**, **piora de latência**, **aumento de erros** e **custo**. Alerta só depois de `patience` dias seguidos com sinal, para não acordar ninguém por ruído.

O foco do projeto não é o painel, é **medir os próprios detectores**: quantos falsos alarmes dão e quantos dias levam para perceber um problema real. Para isso, ele gera fluxos simulados com problemas injetados em dia conhecido.

## Como funciona

```
referência (semana saudável)  ─┐
                               ├─> day_stats  ->  raw (números brutos do dia)
dia atual ─────────────────────┘                    |
                                     flags_from_raw (aplica limiares da config)
                                                    |
                                  alerta só após `patience` dias seguidos
```

| Sinal | Como é decidido |
|---|---|
| Drift de entrada (tokens, score, categoria) | PSI e/ou teste estatístico (KS para números, qui-quadrado para categorias), conforme `method` |
| Qualidade | teste binomial unilateral da taxa de feedback positivo do dia contra a referência |
| Latência | p95 do dia ÷ p95 da referência > 1,25 |
| Erros | teste binomial unilateral contra a taxa de referência |
| Custo | custo médio do dia ÷ da referência > 1,25 |

Separar `raw` de `flags_from_raw` permite **reaplicar outra configuração aos mesmos números** sem recalcular as estatísticas, e é isso que o experimento usa para comparar configurações de forma justa.

## Resultados medidos (simulação, 30 sementes por cenário)

Cada semente: 7 dias de referência e 30 dias de monitoramento, 500 requisições/dia, cerca de 20% delas com feedback. Os limiares (PSI 0,2, alfa 0,001, paciência 2) foram definidos **antes** de rodar e não foram ajustados depois.

**A) Falsos alarmes** (cenário sem problema, 30 sementes × 30 dias)

| Configuração | Sementes com algum alarme falso |
|---|---|
| psi / 1 dia | 7 de 30 |
| psi / 2 dias | 0 de 30 |
| estatístico / 2 dias | 0 de 30 |
| ambos / 2 dias (padrão) | 0 de 30 |

Os 7 alarmes do "1 dia" vieram da **qualidade** (10 dias) e dos **erros** (1 dia), não do PSI: com teste diário a 1%, esperam-se cerca de 9 falsos dias em 900, e foi o que apareceu. A paciência de 2 dias eliminou todos. Atenção: "0 de 30" **não é zero**; com 30 sementes o limite superior plausível ainda é de cerca de 10%.

**B) Detecção** (taxa de detecção / atraso mediano em dias, com p90)

| Cenário (sinal-alvo) | psi / 1 dia | psi / 2 dias | estatístico / 2 | ambos / 2 |
|---|---|---|---|---|
| drift de entrada, tokens ×1,4 | 1,00 / 0 (p90 0) | 1,00 / 1 (p90 1) | 1,00 / 1 (p90 1) | 1,00 / 1 (p90 1) |
| **drift sutil, tokens ×1,15** | 0,00 | 0,00 | **1,00 / 1 (p90 1)** | 0,00 |
| mudança de categorias | 1,00 / 0 | 1,00 / 1 | 1,00 / 1 | 1,00 / 1 |
| **queda de qualidade (0,82 → 0,70)** | 1,00 / 0 (p90 2) | 1,00 / 2 (p90 6) | 1,00 / 2 (p90 6) | 1,00 / 2 (p90 6) |
| latência ×1,5 | 1,00 / 0 | 1,00 / 1 | 1,00 / 1 | 1,00 / 1 |
| erros 1% → 5% | 1,00 / 0 | 1,00 / 1 | 1,00 / 1 | 1,00 / 1 |

A qualidade é a mais lenta e a mais variável (mediana 2 dias, p90 6, máximo 14), porque usa só cerca de 100 feedbacks por dia. Olhar apenas a mediana esconderia isso.

**C) Ponto cego dos monitores de entrada.** Na queda de qualidade com entradas inalteradas, o monitor de qualidade detectou em 100% das sementes e os monitores de drift de entrada em **0%**. Monitorar só a distribuição dos dados não basta, e é preciso medir a qualidade da saída, o que exige algum feedback ou rótulo.

**D) Tráfego alto.** Com 5.000 requisições/dia, uma mudança de apenas +3% nos tokens, sem efeito prático, fez o teste estatístico alertar em **5 de 10 sementes**, enquanto PSI e "ambos" não alertaram nenhuma vez.

## O trade-off que o experimento revela

- **"ambos" (PSI e teste)** é conservador: não dá alarme falso e ignora mudanças pequenas. Em contrapartida, **perdeu o drift sutil de +15%** que o teste estatístico pegou. É o padrão porque, em tráfego alto, evita alertar por diferenças irrelevantes.
- **"estatístico"** é mais sensível, e essa sensibilidade **cresce com o volume**: em 5.000 requisições/dia ele acusa mudanças sem importância (cenário D). Para lotes pequenos e drift que importa mesmo quando é discreto, ele é a melhor escolha.
- Significância estatística não é relevância prática. A escolha entre os dois depende do volume e do custo de um falso alarme, e o experimento serve para decidir com números.

## Como rodar

```bash
pip install -r requirements.txt
pytest -q                     # 16 testes
python eval/avaliar.py 30     # experimento completo (cerca de 80 s), grava results/*.json
```

```python
from modelmonitor.monitor import Monitor, MonitorConfig
m = Monitor(df_referencia, MonitorConfig(method="both", patience=2))
rep = m.observe(df_do_dia)
rep.active_alerts, rep.new_alerts, rep.raw
```

Colunas esperadas: `tokens_prompt`, `score_modelo`, `categoria`, `latency_ms`, `erro` (0/1), `feedback` (1/0 ou NaN), `custo`.

## Limitações

- **Tudo é simulado**, com um cenário de cada tipo. Os números dizem como os detectores se comportam *nessa* simulação, não como se comportariam em um sistema real.
- **Sem sazonalidade**: tráfego real tem ciclos de dia da semana e feriados, que gerariam falsos alarmes com uma referência de uma semana só.
- **Feedback imediato**: na prática, rótulos de qualidade chegam atrasados, o que aumenta o atraso de detecção.
- Os limiares são exemplos razoáveis, não valores de uma empresa real; vale calibrar com dados próprios.
- Uma única fatia de tráfego: sem monitoramento por segmento (idioma, canal, versão do modelo).
- Sem métricas específicas de LLM (toxicidade, alucinação medida por juiz) e sem integração com sistemas de alerta ou painéis.

## Próximos passos

1. Referência sensível a sazonalidade (mesmo dia da semana) e correção para múltiplos testes.
2. Lidar com rótulos atrasados e estimar qualidade por amostragem.
3. Monitorar por segmento e drift de embeddings.
4. Métricas de LLM (taxa de recusa, alucinação por juiz) e integração com Prometheus/Grafana.
5. Validar os detectores em dados reais de produção.
