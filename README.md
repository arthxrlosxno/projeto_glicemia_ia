# Classificacao de Estado Glicemico com CGM (Projeto Final de IA)

Pipeline de **classificacao multiclasse de serie temporal** que preve, com **30
minutos de antecedencia**, em qual estado glicemico um paciente diabetico tipo 1
estara — **hipoglicemia** (`<70`), **normal** (`70-180`) ou **hiperglicemia**
(`>180`) — a partir do historico recente do sensor FreeStyle Libre (CGM). O
projeto compara **3 modelos** (Random Forest, LSTM, GRU) sobre **dados reais de 1
ano**, com 30 execucoes por modelo, teste de hipotese, variacao de
hiperparametros e analise de erro por contexto. Tema alinhado a **ODS 3 - Saude e
Bem-Estar**: antecipar um evento de risco glicemico permite intervir antes do
dano.

> Disciplina de Inteligencia Artificial - UNIFESP - docente Didier A.
> Vega-Oliveros. Autor: Arthur Losano (RA 163564).

---

## 1. Estrutura de pastas

```
projeto_ia_glicemia/
├── data/
│   └── ArthurLosano_glucose_6-10-2026.csv   # CSV bruto do FreeStyle (NAO versionado - ver Sec. 5)
├── src/
│   ├── 01_preprocessing.py        # Fase 1 - limpeza, janelas, features, split, scaler
│   ├── 02_models.py               # Fase 2 - define RF/LSTM/GRU (importado por 03, 05, 07)
│   ├── 03_train_evaluate.py       # Fase 3 - 30 runs por modelo, metricas, tempo
│   ├── 04_statistics.py           # Fase 4 - Wilcoxon pareado + Bonferroni
│   ├── 05_hyperparams.py          # Fase 5 - variacao de units e window (5 runs/valor)
│   ├── 06_plots.py                # Fase 6 - gera todas as figuras
│   └── 07_error_by_context.py     # Fase 7 - erro por contexto (basal/transicao, slope)
├── results/
│   ├── processed/
│   │   ├── dataset.npz             # saida da Fase 1 (X/y, splits, scaler)
│   │   └── predictions_best.npz    # predicoes do melhor run (Fase 3)
│   ├── metrics_30runs.csv          # 90 linhas = 3 modelos x 30 runs (Fase 3)
│   ├── hyperparams_results.csv     # 30 linhas = 6 valores x 5 runs (Fase 5)
│   ├── statistical_tests.txt       # 6 testes Wilcoxon (Fase 4)
│   └── figures/                    # PNGs para slides/relatorio (Fases 6 e 7)
├── requirements.txt
└── README.md
```

---

## 2. Ambiente e instalacao

```powershell
# 1) Criar o venv num caminho CURTO (ver nota MAX_PATH abaixo)
python -m venv C:\glvenv

# 2) Instalar as dependencias
C:\glvenv\Scripts\python -m pip install -r requirements.txt
```

> **Por que `C:\glvenv`?** No Windows, o TensorFlow tem caminhos internos muito
> longos e estoura o limite de 260 caracteres (MAX_PATH) quando o venv fica numa
> pasta profunda

Versoes principais (de `requirements.txt`): pandas 3.0.4, numpy 2.4.6,
scikit-learn 1.9.0, scipy 1.17.1, matplotlib 3.11.0, **tensorflow-cpu 2.21.0**
(Keras 3.x embutido).

---

## 3. Como rodar (ordem linear `01` -> `07`)

Executar cada script com o Python do venv, **na ordem**. Cada script imprime um
"RESUMO DA FASE" com os pontos de validacao.

```powershell
C:\glvenv\Scripts\python.exe src\01_preprocessing.py     # gera results/processed/dataset.npz
C:\glvenv\Scripts\python.exe src\02_models.py            # smoke test dos 3 modelos (rapido)
C:\glvenv\Scripts\python.exe src\03_train_evaluate.py    # 30 runs/modelo -> metrics_30runs.csv + predictions_best.npz
C:\glvenv\Scripts\python.exe src\04_statistics.py        # Wilcoxon -> statistical_tests.txt
C:\glvenv\Scripts\python.exe src\05_hyperparams.py       # variacao de units/window -> hyperparams_results.csv
C:\glvenv\Scripts\python.exe src\06_plots.py             # gera todas as figuras em results/figures/
C:\glvenv\Scripts\python.exe src\07_error_by_context.py  # erro por regime -> 2 figuras (sem retreino)
```

**Custo de tempo:**

| Script | Tempo | Observacao |
|---|---|---|
| 01, 02, 04, 06, 07 | segundos a ~1 min | rapidos |
| **03** (30 runs x 3 modelos) | **~2h07** | caro; **retomavel** |
| **05** (30 treinos de LSTM) | **~33 min** | caro; **retomavel** |

> **Retomavel (03 e 05):** cada run e gravado no CSV assim que termina; ao
> reiniciar o script, os runs ja presentes sao pulados. Para refazer do zero,
> apague o CSV correspondente (e `predictions_best.npz`, no caso do 03) antes.

Validacao rapida (sem rodar o treino completo): `03` e `05` aceitam dois
argumentos `n_runs n_epocas`, ex.: `... src\03_train_evaluate.py 2 3`.

---

## 4. Definicao do problema

- **Entrada:** janela de 8 leituras (2h de historico), como serie bruta + features
  derivadas (valor atual, media, desvio, slope, derivada, hora em sin/cos).
- **Alvo (y):** classe da glicose em **t+2 leituras (30 min a frente)** - hipo
  (`0`), normal (`1`), hiper (`2`).
- **Split cronologico 80/20** (nunca embaralhar - evita vazamento temporal).
- **Scaler ajustado so no treino.** Desbalanceamento (hipo ~1,4%) tratado com
  `class_weight`.

---

## 5. Dados de entrada

- **Arquivo:** `data/ArthurLosano_glucose_6-10-2026.csv` (export do app FreeStyle
  LibreLink). Ler com `skiprows=1` (a 1a linha e metadado); usar so `Tipo de
  registro == 0` (leitura automatica do sensor).
- **Escopo:** ultimos **12 meses** (~34.089 leituras, frequencia 15 min). Um ano
  da ~140 hipos no teste, tornando o recall-hipo uma metrica estavel.
- **Privacidade:** o CSV contem dados pessoais de saude e **nao e versionado no
  repositorio**. Para reproduzir, coloque o arquivo em `data/` com esse nome.
  Sem o CSV, o `01_preprocessing.py` para e avisa.

---

## 6. Principais resultados

Medias de **30 runs** por modelo (Fase 3):

| Modelo | F1-macro | recall-hipo | tempo/treino |
|---|---|---|---|
| Random Forest | **0,802** | 0,561 | ~1,7 s |
| LSTM | 0,749 | **0,936** | ~134 s |
| GRU | 0,753 | **0,926** | ~115 s |

**Trade-off clinico (comprovado por Wilcoxon pareado + Bonferroni, Fase 4):** o
Random Forest e significativamente melhor em **F1-macro** (p ~ 1e-9), enquanto
LSTM e GRU sao significativamente melhores em **recall-hipo** (p ~ 1e-6) - sinais
opostos conforme a metrica. LSTM e GRU sao estatisticamente equivalentes entre
si. Conclusao: **nao existe um unico "melhor modelo"** - depende do objetivo
clinico (alerta de hipo vs. classificacao global).

**Achado da Fase 7:** **130 de 140 hipos (93%) ocorrem no regime BASAL** (baixa
variabilidade - quedas graduais de madrugada/jejum), nao em transicoes bruscas. O
periodo "calmo" esconde o maior risco clinico, e e justamente onde as redes
recorrentes superam o RF.

---

## 7. Figuras (`results/figures/`)

| Figura | Conteudo |
|---|---|
| `06_distribuicao_classes.png` | Distribuicao das 3 classes (evidencia o desbalanceamento) |
| `06_exemplo_janela.png` | Exemplo de janela de 8 leituras + alvo t+2 destacado |
| `06_f1_por_modelo.png` | F1-macro medio +/- desvio dos 3 modelos |
| `06_tradeoff_f1_vs_recallhipo.png` | **Trade-off** F1-macro vs recall-hipo por modelo (grafico-sintese) |
| `06_matrizes_confusao.png` | Matrizes de confusao do melhor run (ilustrativo) dos 3 modelos |
| `06_f1_vs_hiperparametro.png` | F1-macro vs LSTM units e vs window size (Fase 5) |
| `06_tempo_execucao.png` | Tempo medio de treino por modelo |
| `07_erro_por_variabilidade.png` | F1 e recall-hipo por regime basal vs transicao |
| `07_erro_por_slope.png` | F1 e recall-hipo por direcao do slope (descendo/estavel/subindo) |

---
