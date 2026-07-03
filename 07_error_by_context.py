# -*- coding: utf-8 -*-
"""
FASE 7 - Analise de Erro por Contexto Temporal
==============================================

Objetivo: mostrar, com os proprios dados, onde o modelo erra e por que -
sustentando a tese de que o gargalo e a informacao ausente (insulina,
refeicao, exercicio), nao a arquitetura.

"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, recall_score

# --------------------------------------------------------------------------
# Caminhos e artefatos (so leitura - nada de treino)
# --------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_DIR / "results"
FIG_DIR = RESULTS_DIR / "figures"
DATASET = RESULTS_DIR / "processed" / "dataset.npz"
PRED = RESULTS_DIR / "processed" / "predictions_best.npz"

# Indices das features brutas (conferidos: ver feature_names da Fase 1).
IDX_DESVIO = 10
IDX_SLOPE = 11

MODELOS = ["RandomForest", "LSTM", "GRU"]
PREFIXO = {"RandomForest": "rf", "LSTM": "lstm", "GRU": "gru"}
COR_MODELO = {"RandomForest": "#7a7acc", "LSTM": "#2b6cb0", "GRU": "#2a9d8f"}


def estilo_eixo(ax):
    ax.set_facecolor("white")
    ax.grid(True, linestyle="--", alpha=0.4)


def metricas_no_recorte(y_true, y_pred, mascara):
    """F1-macro e recall-hipo (classe 0) calculados somente nas amostras da mascara."""
    yt = y_true[mascara]
    yp = y_pred[mascara]
    f1 = f1_score(yt, yp, labels=[0, 1, 2], average="macro", zero_division=0)
    recall_hipo = recall_score(yt, yp, labels=[0, 1, 2], average=None,
                               zero_division=0)[0]
    return f1, recall_hipo


def construir_regimes(desvio_treino, desvio_teste, slope_treino, slope_teste, y_test):
    """Monta as mascaras das duas segmentacoes. Limiar SEMPRE do treino."""
    # (a) Variabilidade: mediana do desvio no TREINO.
    med_desvio = float(np.median(desvio_treino))
    basal = desvio_teste <= med_desvio          # baixa variabilidade
    transicao = ~basal                           # alta variabilidade

    # (b) Direcao do slope: terciles do slope no TREINO.
    q1, q2 = np.quantile(slope_treino, [1 / 3, 2 / 3])
    descendo = slope_teste <= q1                 # queda (risco de hipo)
    subindo = slope_teste >= q2                  # subida
    estavel = ~(descendo | subindo)              # meio

    variabilidade = {"basal": basal, "transicao": transicao}
    slope = {"descendo": descendo, "estavel": estavel, "subindo": subindo}
    limiares = {"mediana_desvio_treino": med_desvio,
                "slope_q1_treino": float(q1), "slope_q2_treino": float(q2)}
    return variabilidade, slope, limiares


def imprimir_distribuicao_hipo(y_test, regimes, nome_seg):
    """Mostra quantas HIPOS reais caem em cada regime."""
    print("\n[Distribuicao das HIPOS reais por regime - %s]" % nome_seg)
    total_hipo = int((y_test == 0).sum())
    soma = 0
    for nome, masc in regimes.items():
        n_amostras = int(masc.sum())
        n_hipo = int(((y_test == 0) & masc).sum())
        soma += n_amostras
        print("    %-10s amostras=%5d | hipos=%3d (%.0f%% das hipos)"
              % (nome, n_amostras, n_hipo, 100.0 * n_hipo / total_hipo))
    print("    soma das amostras=%d | tamanho do teste=%d | iguais=%s"
          % (soma, len(y_test), soma == len(y_test)))


def tabela_metricas(y_test, preds, regimes, nome_seg):
    """Imprime F1-macro e recall-hipo por modelo x regime e devolve os numeros
    num dicionario para o grafico."""
    print("\n[%s] F1-macro / recall-hipo por modelo x regime:" % nome_seg)
    cabecalho = "    %-13s" % "modelo" + "".join("| %-22s" % r for r in regimes)
    print(cabecalho)
    resultado = {}  # resultado[modelo][regime] = (f1, recall_hipo)
    for nome in MODELOS:
        y_pred = preds[PREFIXO[nome] + "_pred"]
        resultado[nome] = {}
        linha = "    %-13s" % nome
        for regime, masc in regimes.items():
            f1, rec = metricas_no_recorte(y_test, y_pred, masc)
            resultado[nome][regime] = (f1, rec)
            linha += "| F1=%.3f rec-hipo=%.3f " % (f1, rec)
        print(linha)
    return resultado


def grafico_por_regime(resultado, regimes, nome_seg, nome_arquivo, titulo):
    """Barras agrupadas: dois paineis (F1-macro e recall-hipo). Em cada painel,
    grupos = regimes, barras = modelos."""
    nomes_regime = list(regimes.keys())
    fig, eixos = plt.subplots(1, 2, figsize=(12, 4.8), dpi=150)

    for ax, idx_metrica, titulo_painel in [
        (eixos[0], 0, "F1-macro por regime"),
        (eixos[1], 1, "recall-hipo por regime"),
    ]:
        x = np.arange(len(nomes_regime))
        largura = 0.26
        for j, nome in enumerate(MODELOS):
            valores = [resultado[nome][r][idx_metrica] for r in nomes_regime]
            ax.bar(x + (j - 1) * largura, valores, largura,
                   color=COR_MODELO[nome], label=nome)
        ax.set_xticks(x)
        ax.set_xticklabels(nomes_regime)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("metrica (melhor run por F1-macro)")
        ax.set_title(titulo_painel)
        estilo_eixo(ax)
        ax.legend(fontsize=8)

    fig.suptitle(titulo)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    caminho = FIG_DIR / nome_arquivo
    fig.savefig(caminho, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return caminho


def main():
    print("Carregando artefatos (sem retreinar)...")
    d = np.load(DATASET, allow_pickle=True)
    p = np.load(PRED, allow_pickle=True)

    y_test = p["y_test"]
    # Sanity: o y_test das predicoes deve ser o mesmo da Fase 1.
    assert np.array_equal(y_test, d["y_test"]), "y_test das predicoes != dataset"

    desvio_treino = d["X_tab_raw_train"][:, IDX_DESVIO]
    desvio_teste = d["X_tab_raw_test"][:, IDX_DESVIO]
    slope_treino = d["X_tab_raw_train"][:, IDX_SLOPE]
    slope_teste = d["X_tab_raw_test"][:, IDX_SLOPE]

    variabilidade, slope, limiares = construir_regimes(
        desvio_treino, desvio_teste, slope_treino, slope_teste, y_test)

    print("\n" + "=" * 78)
    print("RESUMO DA FASE 7 - ERRO POR CONTEXTO TEMPORAL")
    print("=" * 78)
    print("\n[Limiares calculados NO TREINO (sem vazar o teste)]")
    print("    mediana do desvio (basal/transicao) = %.3f" % limiares["mediana_desvio_treino"])
    print("    terciles do slope (descendo/estavel/subindo) = %.3f , %.3f"
          % (limiares["slope_q1_treino"], limiares["slope_q2_treino"]))

    # --- Segmentacao (a): variabilidade ---
    imprimir_distribuicao_hipo(y_test, variabilidade, "variabilidade (basal/transicao)")
    res_var = tabela_metricas(y_test, p, variabilidade, "VARIABILIDADE")

    # --- Segmentacao (b): direcao do slope ---
    imprimir_distribuicao_hipo(y_test, slope, "direcao do slope")
    res_slope = tabela_metricas(y_test, p, slope, "SLOPE")

    # --- Checkpoint do achado central ---
    n_hipo_basal = int(((y_test == 0) & variabilidade["basal"]).sum())
    n_hipo_total = int((y_test == 0).sum())
    achado_ok = n_hipo_basal >= 0.8 * n_hipo_total  # esperado ~130/140
    print("\n[Checkpoint do achado central]")
    print("    hipos no BASAL = %d/%d (%.0f%%) | esperado ~130/140 | OK=%s"
          % (n_hipo_basal, n_hipo_total, 100.0 * n_hipo_basal / n_hipo_total, achado_ok))
    if not achado_ok:
        print("    ** ATENCAO: hipos NAO concentradas no basal - segmentacao pode estar")
        print("       invertida. Revisar antes de usar no relatorio. **")

    # --- Graficos ---
    fig_var = grafico_por_regime(
        res_var, variabilidade, "variabilidade",
        "07_erro_por_variabilidade.png",
        "Fase 7 - Erro por variabilidade (basal vs transicao)\n"
        "recall-hipo e baixo no RF mesmo no BASAL: o periodo 'calmo' esconde o risco")
    fig_slope = grafico_por_regime(
        res_slope, slope, "slope",
        "07_erro_por_slope.png",
        "Fase 7 - Erro por direcao do slope (descendo/estavel/subindo)\n"
        "a hipo concentra-se em 'descendo' (quedas graduais), nao em transicoes bruscas")

    print("\n[Graficos salvos]")
    print("    %s" % fig_var)
    print("    %s" % fig_slope)
    print("\n[Honestidade] metricas por regime usam o melhor run POR F1-MACRO")
    print("    (ilustrativo). A captura de hipo TIPICA e a media das 30 runs da")
    print("    Fase 3 (LSTM 0,936 / GRU 0,926 / RF 0,561).")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
