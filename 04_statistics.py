# -*- coding: utf-8 -*-
"""
FASE 4 - Analise Estatistica (Wilcoxon pareado)
===============================================
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
CSV_IN = PROJECT_DIR / "results" / "metrics_30runs.csv"
TXT_OUT = PROJECT_DIR / "results" / "statistical_tests.txt"

ALPHA = 0.05
N_TESTES = 6                       # 3 pares x 2 metricas
ALPHA_BONF = ALPHA / N_TESTES      # ≈ 0,0083

METRICAS = ["f1_macro", "recall_hipo"]
PARES = [("RandomForest", "LSTM"), ("RandomForest", "GRU"), ("LSTM", "GRU")]


def serie_por_run(df, modelo, metrica):
    """Devolve os valores da metrica para um modelo, ordenados por run, para que
    o pareamento (run i de A com run i de B) seja consistente."""
    sub = df[df["modelo"] == modelo].sort_values("run")
    return sub[metrica].values


def testar_par(df, modelo_a, modelo_b, metrica):
    """Roda o Wilcoxon pareado entre dois modelos numa metrica e devolve um dict
    com medias, diferenca, p-valor e conclusoes (com e sem Bonferroni)."""
    a = serie_por_run(df, modelo_a, metrica)
    b = serie_por_run(df, modelo_b, metrica)

    media_a, media_b = a.mean(), b.mean()
    diff = media_a - media_b
    melhor = modelo_a if diff > 0 else modelo_b

    # Se as duas series forem identicas, o Wilcoxon nao se aplica (sem diferenca).
    try:
        _, p = wilcoxon(a, b)
    except ValueError:
        p = float("nan")

    return {
        "metrica": metrica,
        "par": "%s vs %s" % (modelo_a, modelo_b),
        "media_a": media_a, "media_b": media_b, "diff": diff,
        "melhor": melhor,
        "p": p,
        "sig_005": (p < ALPHA) if not np.isnan(p) else False,
        "sig_bonf": (p < ALPHA_BONF) if not np.isnan(p) else False,
    }


def formatar_linhas(resultados):
    """Monta as linhas de texto do relatorio estatistico."""
    linhas = []
    linhas.append("=" * 74)
    linhas.append("FASE 4 - TESTES DE HIPOTESE (Wilcoxon pareado)")
    linhas.append("=" * 74)
    linhas.append("alpha = %.3f | testes = %d | alpha Bonferroni = %.4f" % (ALPHA, N_TESTES, ALPHA_BONF))
    linhas.append("Conclusao usa o alpha de Bonferroni (corrige multiplas comparacoes).")

    for metrica in METRICAS:
        linhas.append("")
        linhas.append("-" * 74)
        linhas.append("METRICA: %s" % metrica)
        linhas.append("-" * 74)
        for r in [x for x in resultados if x["metrica"] == metrica]:
            if np.isnan(r["p"]):
                concl = "indefinido (series identicas)"
            elif r["sig_bonf"]:
                concl = "SIGNIFICATIVO (Bonferroni): %s e melhor" % r["melhor"]
            elif r["sig_005"]:
                concl = "significativo so a 0,05, NAO apos Bonferroni -> tratar como equivalente"
            else:
                concl = "NAO significativo -> modelos equivalentes nesta metrica"
            linhas.append("  %-26s media: %.4f vs %.4f (dif=%+.4f) | p=%.2e" %
                          (r["par"], r["media_a"], r["media_b"], r["diff"], r["p"]))
            linhas.append("        -> %s" % concl)

    return linhas


def main():
    print("Carregando metricas da Fase 3...")
    df = pd.read_csv(CSV_IN)

    resultados = []
    for metrica in METRICAS:
        for (a, b) in PARES:
            resultados.append(testar_par(df, a, b, metrica))

    linhas = formatar_linhas(resultados)

    # Salva o relatorio em txt.
    TXT_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(TXT_OUT, "w") as f:
        f.write("\n".join(linhas) + "\n")

    # Imprime na tela (modo incremental) + RESUMO DA FASE com checkpoints.
    print("\n".join(linhas))

    print("\n" + "=" * 74)
    print("RESUMO DA FASE 4 - checkpoints")
    print("=" * 74)
    ps = [r["p"] for r in resultados if not np.isnan(r["p"])]
    print("  p-valores entre 0 e 1: %s" % all(0.0 <= p <= 1.0 for p in ps))
    # Coerencia com a Fase 3: em F1-macro o RF deve vencer; em recall-hipo, perder.
    f1_rf_lstm = next(r for r in resultados if r["metrica"] == "f1_macro" and r["par"] == "RandomForest vs LSTM")
    rec_rf_lstm = next(r for r in resultados if r["metrica"] == "recall_hipo" and r["par"] == "RandomForest vs LSTM")
    print("  Sinais opostos esperados (RF melhor em F1, pior em recall-hipo): %s"
          % (f1_rf_lstm["diff"] > 0 and rec_rf_lstm["diff"] < 0))
    print("  Arquivo salvo em: %s" % TXT_OUT)
    print("=" * 74 + "\n")


if __name__ == "__main__":
    main()
