# -*- coding: utf-8 -*-
"""
FASE 6 - Graficos para Slides e Relatorio
==========================================

Gera as figuras finais a partir dos artefatos JA salvos pelas fases anteriores:

  - results/processed/dataset.npz       (Fase 1: y, janelas, timestamps)
  - results/metrics_30runs.csv          (Fase 3: 30 runs por modelo)
  - results/processed/predictions_best.npz (Fase 3: predicoes do melhor run)
  - results/hyperparams_results.csv     (Fase 5: F1 vs hiperparametro)

"""

import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# --------------------------------------------------------------------------
# Caminhos e artefatos
# --------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_DIR / "results"
FIG_DIR = RESULTS_DIR / "figures"
DATASET = RESULTS_DIR / "processed" / "dataset.npz"
PRED = RESULTS_DIR / "processed" / "predictions_best.npz"
METRICS_CSV = RESULTS_DIR / "metrics_30runs.csv"
HP_CSV = RESULTS_DIR / "hyperparams_results.csv"
DATA_CSV = PROJECT_DIR / "data" / "ArthurLosano_glucose_6-10-2026.csv"

# Ordem e rotulos fixos para todos os graficos.
MODELOS = ["RandomForest", "LSTM", "GRU"]
CLASSES = ["hipo", "normal", "hiper"]
COR_CLASSE = {"hipo": "#d64545", "normal": "#3a9d5d", "hiper": "#e0962f"}
COR_MODELO = {"RandomForest": "#7a7acc", "LSTM": "#2b6cb0", "GRU": "#2a9d8f"}

# Limiares clinicos (mesmos da Fase 1): hipo < 70, normal 70-180, hiper > 180.
LIMIAR_HIPO = 70
LIMIAR_HIPER = 180

# Window/horizon do projeto (para o grafico de exemplo de janela).
WINDOW = 8
HORIZON = 2


def estilo_eixo(ax):
    """Aplica o estilo padrao do projeto: fundo branco, grid suave tracejado."""
    ax.set_facecolor("white")
    ax.grid(True, linestyle="--", alpha=0.4)


def salvar(fig, nome):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    caminho = FIG_DIR / nome
    fig.savefig(caminho, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return caminho


# --------------------------------------------------------------------------
# 1) Distribuicao das classes (evidencia o desbalanceamento)
# --------------------------------------------------------------------------
def grafico_distribuicao(d):
    y_train, y_test = d["y_train"], d["y_test"]
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)

    cont_tr = [int((y_train == k).sum()) for k in (0, 1, 2)]
    cont_te = [int((y_test == k).sum()) for k in (0, 1, 2)]
    x = np.arange(3)
    largura = 0.38

    b1 = ax.bar(x - largura / 2, cont_tr, largura, label="treino", color="#9aa8c7")
    b2 = ax.bar(x + largura / 2, cont_te, largura, label="teste", color="#2b6cb0")

    # Percentual em cima de cada barra (o desbalanceamento fica visivel).
    for barras, cont in [(b1, cont_tr), (b2, cont_te)]:
        total = sum(cont)
        for bar, c in zip(barras, cont):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    "%.1f%%" % (100.0 * c / total), ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(["%s (%d)" % (c, k) for k, c in enumerate(CLASSES)])
    ax.set_ylabel("numero de janelas")
    ax.set_title("Distribuicao das classes - alvo em t+2 (30 min)\n"
                 "hipo e classe rara (~1,4%): desbalanceamento tratado com class_weight")
    ax.legend()
    estilo_eixo(ax)
    return salvar(fig, "06_distribuicao_classes.png")


# --------------------------------------------------------------------------
# 2) Exemplo de janela com a leitura-alvo destacada (didatico)
# --------------------------------------------------------------------------
def carregar_modulo_preproc():
    spec = importlib.util.spec_from_file_location("preproc", SCRIPT_DIR / "01_preprocessing.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def achar_janela_didatica(g, ts, gap_max_min):
    """Procura uma janela limpa (sem lacuna) cujo alvo t+2 seja HIPO"""
    fallback = None
    ultimo = len(g) - WINDOW - HORIZON
    for i in range(ultimo + 1):
        idx_alvo = i + WINDOW + HORIZON - 1
        trecho = ts[i: idx_alvo + 1]
        difs = np.diff(trecho).astype("timedelta64[m]").astype(int)
        if (difs > gap_max_min).any():
            continue
        if fallback is None:
            fallback = i
        if g[idx_alvo] < LIMIAR_HIPO:  # alvo e hipo
            return i
    return fallback


def grafico_exemplo_janela(P):
    df = P.carregar_e_limpar(DATA_CSV)
    df_ano = P.recortar_periodo(df, P.DIAS_RECORTE)
    g = df_ano["g"].values
    ts = df_ano["ts"].values

    i = achar_janela_didatica(g, ts, P.GAP_MAX_MIN)
    idx_alvo = i + WINDOW + HORIZON - 1
    janela = g[i: i + WINDOW]
    alvo = g[idx_alvo]

    # eixos x em minutos relativos: entrada -105..0, alvo +30 (15 min/leitura).
    x_entrada = np.arange(-(WINDOW - 1), 1) * 15
    x_alvo = HORIZON * 15

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)

    # Faixas clinicas de fundo.
    ax.axhspan(0, LIMIAR_HIPO, color=COR_CLASSE["hipo"], alpha=0.10)
    ax.axhspan(LIMIAR_HIPO, LIMIAR_HIPER, color=COR_CLASSE["normal"], alpha=0.10)
    ax.axhspan(LIMIAR_HIPER, max(janela.max(), alvo) + 40,
               color=COR_CLASSE["hiper"], alpha=0.10)
    ax.axhline(LIMIAR_HIPO, color=COR_CLASSE["hipo"], linestyle=":", linewidth=1)
    ax.axhline(LIMIAR_HIPER, color=COR_CLASSE["hiper"], linestyle=":", linewidth=1)

    # Entrada (8 leituras) e alvo (t+2).
    ax.plot(x_entrada, janela, "-o", color="#2b6cb0", label="entrada: 8 leituras (2h)")
    # linha pontilhada ligando a ultima leitura ao alvo (os 30 min a prever).
    ax.plot([0, x_alvo], [janela[-1], alvo], "--", color="#888888")
    classe_alvo = CLASSES[0 if alvo < LIMIAR_HIPO else (1 if alvo <= LIMIAR_HIPER else 2)]
    ax.plot([x_alvo], [alvo], "P", markersize=14, color=COR_CLASSE[classe_alvo],
            label="alvo t+2 (+30 min): %.0f mg/dL = %s" % (alvo, classe_alvo))

    ax.set_xlabel("minutos relativos a leitura atual (0 = agora)")
    ax.set_ylabel("glicose (mg/dL)")
    ax.set_title("Exemplo de janela: prever o estado glicemico 30 min a frente\n"
                 "faixas: hipo <70  |  normal 70-180  |  hiper >180")
    ax.legend(loc="best", fontsize=8)
    estilo_eixo(ax)
    return salvar(fig, "06_exemplo_janela.png")


# --------------------------------------------------------------------------
# 3) F1-macro medio +/- desvio dos 3 modelos
# --------------------------------------------------------------------------
def grafico_f1_por_modelo(m):
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    medias, desvios = [], []
    for nome in MODELOS:
        sub = m[m["modelo"] == nome]
        medias.append(sub["f1_macro"].mean())
        desvios.append(sub["f1_macro"].std())

    x = np.arange(len(MODELOS))
    barras = ax.bar(x, medias, yerr=desvios, capsize=6,
                    color=[COR_MODELO[n] for n in MODELOS], alpha=0.9)
    for bar, mu, sd in zip(barras, medias, desvios):
        ax.text(bar.get_x() + bar.get_width() / 2, mu + sd + 0.005,
                "%.3f\n+/-%.3f" % (mu, sd), ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(MODELOS)
    ax.set_ylabel("F1-macro (media +/- desvio, 30 runs)")
    ax.set_ylim(0, 1.0)
    ax.set_title("F1-macro por modelo - 30 runs com sementes diferentes\n"
                 "RF lidera em F1-macro (metrica global equilibrada)")
    estilo_eixo(ax)
    return salvar(fig, "06_f1_por_modelo.png")


# --------------------------------------------------------------------------
# 3b) Trade-off F1-macro vs recall-hipo (GRAFICO PRINCIPAL DOS SLIDES)
# --------------------------------------------------------------------------
def grafico_tradeoff(m):
    """Barras agrupadas: para cada modelo, F1-macro e recall-hipo lado a lado
    (media +/- desvio das 30 runs)."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    f1_mu = [m[m["modelo"] == n]["f1_macro"].mean() for n in MODELOS]
    f1_sd = [m[m["modelo"] == n]["f1_macro"].std() for n in MODELOS]
    rec_mu = [m[m["modelo"] == n]["recall_hipo"].mean() for n in MODELOS]
    rec_sd = [m[m["modelo"] == n]["recall_hipo"].std() for n in MODELOS]

    x = np.arange(len(MODELOS))
    largura = 0.38
    # Duas metricas com cores bem distintas (legibilidade para slide).
    b1 = ax.bar(x - largura / 2, f1_mu, largura, yerr=f1_sd, capsize=5,
                color="#3a6ea5", label="F1-macro (metrica global)")
    b2 = ax.bar(x + largura / 2, rec_mu, largura, yerr=rec_sd, capsize=5,
                color="#d6692f", label="recall-hipo (valor clinico)")

    for barras, mus, sds in [(b1, f1_mu, f1_sd), (b2, rec_mu, rec_sd)]:
        for bar, mu, sd in zip(barras, mus, sds):
            ax.text(bar.get_x() + bar.get_width() / 2, mu + sd + 0.01,
                    "%.3f" % mu, ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(MODELOS)
    ax.set_ylabel("metrica (media +/- desvio, 30 runs)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Trade-off clinico: nao existe um unico 'melhor modelo'\n"
                 "RF vence em F1-macro; LSTM/GRU vencem em recall-hipo "
                 "(ambos significativos, Fase 4)")
    ax.legend(loc="lower center")
    estilo_eixo(ax)
    return salvar(fig, "06_tradeoff_f1_vs_recallhipo.png")


# --------------------------------------------------------------------------
# 4) Matrizes de confusao do melhor run (ILUSTRATIVO) dos 3 modelos
# --------------------------------------------------------------------------
def grafico_matrizes_confusao(p, m):
    y_test = p["y_test"]
    prefixo = {"RandomForest": "rf", "LSTM": "lstm", "GRU": "gru"}

    fig, eixos = plt.subplots(1, 3, figsize=(13, 4.5), dpi=150)
    for ax, nome in zip(eixos, MODELOS):
        sub = m[m["modelo"] == nome]
        melhor_run = int(sub.loc[sub["f1_macro"].idxmax(), "run"])
        y_pred = p[prefixo[nome] + "_pred"]
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(CLASSES); ax.set_yticklabels(CLASSES)
        ax.set_xlabel("previsto"); ax.set_ylabel("real")
        # Numero em cada celula (texto claro no fundo escuro e vice-versa).
        vmax = cm.max()
        for i in range(3):
            for j in range(3):
                cor = "white" if cm[i, j] > vmax * 0.5 else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=cor, fontsize=10)
        ax.set_title("%s\n(melhor run por F1-macro = run %d; ILUSTRATIVO)"
                     % (nome, melhor_run), fontsize=9)

    fig.suptitle("Matrizes de confusao (linhas=real, colunas=previsto) - "
                 "run ilustrativo, nao o caso tipico\n"
                 "captura de hipo tipica (media 30 runs): LSTM 0,936  GRU 0,926  RF 0,561",
                 fontsize=10)
    fig.tight_layout()
    return salvar(fig, "06_matrizes_confusao.png")


# --------------------------------------------------------------------------
# 5) F1 vs hiperparametro (reaproveita os dados da Fase 5)
# --------------------------------------------------------------------------
def grafico_f1_vs_hiperparametro(hp):
    fig, eixos = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)
    n_runs = int(hp.groupby(["experimento", "valor"]).size().max())

    for ax, exp, xlabel in [
        (eixos[0], "units", "LSTM units"),
        (eixos[1], "window", "Window size (leituras)"),
    ]:
        sub = hp[hp["experimento"] == exp]
        valores = sorted(sub["valor"].unique())
        medias = [sub[sub["valor"] == v]["f1_macro"].mean() for v in valores]
        desvios = [sub[sub["valor"] == v]["f1_macro"].std() for v in valores]
        ax.errorbar(valores, medias, yerr=desvios, marker="o", capsize=5,
                    color="#2b6cb0", ecolor="#90a4ba", linewidth=2)
        ax.set_xticks(valores)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("F1-macro (media +/- desvio, %d runs)" % n_runs)
        ax.set_title("F1-macro vs %s" % xlabel)
        estilo_eixo(ax)

    fig.suptitle("Influencia dos hiperparametros (LSTM) - Fase 5\n"
                 "capacidade (units) e plana; mais historico (window) ajuda modestamente")
    fig.tight_layout()
    return salvar(fig, "06_f1_vs_hiperparametro.png")


# --------------------------------------------------------------------------
# 6) Tempo de execucao por modelo (opcional)
# --------------------------------------------------------------------------
def grafico_tempo(m):
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    medias = [m[m["modelo"] == n]["tempo_s"].mean() for n in MODELOS]
    desvios = [m[m["modelo"] == n]["tempo_s"].std() for n in MODELOS]
    x = np.arange(len(MODELOS))
    barras = ax.bar(x, medias, yerr=desvios, capsize=6,
                    color=[COR_MODELO[n] for n in MODELOS], alpha=0.9)
    for bar, mu in zip(barras, medias):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                "%.1f s" % mu, ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(MODELOS)
    ax.set_ylabel("tempo medio de treino (s, 30 runs)")
    ax.set_title("Tempo de execucao por modelo (treino)\n"
                 "RF treina em segundos; LSTM/GRU sao ~70-130x mais lentos")
    estilo_eixo(ax)
    return salvar(fig, "06_tempo_execucao.png")


def main():
    print("Carregando artefatos (sem retreinar)...")
    d = np.load(DATASET, allow_pickle=True)
    p = np.load(PRED, allow_pickle=True)
    m = pd.read_csv(METRICS_CSV)
    hp = pd.read_csv(HP_CSV, dtype={"valor": str})
    P = carregar_modulo_preproc()

    print("\n" + "=" * 70)
    print("RESUMO DA FASE 6 - GRAFICOS")
    print("=" * 70)

    figuras = []
    figuras.append(("distribuicao das classes", grafico_distribuicao(d)))
    figuras.append(("exemplo de janela + alvo", grafico_exemplo_janela(P)))
    figuras.append(("F1-macro por modelo", grafico_f1_por_modelo(m)))
    figuras.append(("trade-off F1 vs recall-hipo", grafico_tradeoff(m)))
    figuras.append(("matrizes de confusao", grafico_matrizes_confusao(p, m)))
    figuras.append(("F1 vs hiperparametro", grafico_f1_vs_hiperparametro(hp)))
    figuras.append(("tempo de execucao", grafico_tempo(m)))

    print("\n%d figuras geradas em %s:" % (len(figuras), FIG_DIR))
    for desc, caminho in figuras:
        print("  - %-26s -> %s" % (desc, caminho.name))

    # Numeros-chave reaproveitados (para conferir contra as fases anteriores).
    print("\n[Checkpoints - numeros reaproveitados, nao recalculados pelo modelo]")
    for nome in MODELOS:
        sub = m[m["modelo"] == nome]
        print("  %-13s F1=%.3f+/-%.3f | recall-hipo=%.3f+/-%.3f | tempo=%.1fs"
              % (nome, sub["f1_macro"].mean(), sub["f1_macro"].std(),
                 sub["recall_hipo"].mean(), sub["recall_hipo"].std(),
                 sub["tempo_s"].mean()))
    print("  Matrizes = melhor run POR F1-MACRO (ilustrativo); narrativa de hipo")
    print("  ancorada na MEDIA das 30 runs (LSTM 0,936 / GRU 0,926).")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
