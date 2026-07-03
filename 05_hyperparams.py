# -*- coding: utf-8 -*-
"""
FASE 5 - Variacao de Hiperparametros
=====================================

Mostra como o F1-macro da LSTM reage
a variacao de dois hiperparametros, um por vez, mantendo o MESMO split
cronologico e os EPOCHS fixos da Fase 3.

  A) LSTM_UNITS  {32, 64, 128}  -> nao reprocessa dados (usa o dataset.npz da
     Fase 1, window=8). Mais barato; roda primeiro para validar o script.
  B) WINDOW_SIZE {4, 8, 16}     -> reprocessa os dados (re-gera as janelas com
     o MESMO tratamento de lacuna temporal da Fase 1 - gap > 30 min descartado
     do inicio da janela ATE o alvo t+2, nao so dentro da janela de entrada).

Cada valor de parametro usa N_RUNS_HP=5 sementes diferentes (nao 30, nao 1):
1 run seria ruido (a LSTM varia +/-0,024 entre sementes na Fase 3); 5 runs da
media +/- desvio (barras de erro honestas) sem o custo de 30 runs.

Modelo usado: LSTM (a rede recorrente principal do projeto). O ponto
units=64 / window=8 e o mesmo experimento em ambas as curvas e serve de
SANITY CHECK CRUZADO: deve ficar perto do F1=0,749 que a LSTM teve na Fase 3
(rodada com 30 sementes). Se divergir muito, o setup da Fase 5 nao esta
consistente com o resto do pipeline.

RETOMAVEL: cada combinacao (experimento, valor, run) e gravada no CSV assim
que termina; ao reiniciar, combinacoes ja feitas sao puladas (mesmo padrao da
Fase 3).

"""

import csv
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import tensorflow as tf

# --------------------------------------------------------------------------
# Caminhos e import dos modulos numerados (nao importaveis pelo nome padrao
# por comecarem com digito - mesmo padrao usado em 03_train_evaluate.py).
# --------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATASET = PROJECT_DIR / "results" / "processed" / "dataset.npz"
DATA_CSV = PROJECT_DIR / "data" / "ArthurLosano_glucose_6-10-2026.csv"
RESULTS_DIR = PROJECT_DIR / "results"
CSV_OUT = RESULTS_DIR / "hyperparams_results.csv"


def carregar_modulo(nome_arquivo, nome_modulo):
    spec = importlib.util.spec_from_file_location(nome_modulo, SCRIPT_DIR / nome_arquivo)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = carregar_modulo("01_preprocessing.py", "preprocessamento")
M = carregar_modulo("02_models.py", "modelos")

N_RUNS_HP = 5
EPOCHS = M.EPOCHS
UNITS_VALUES = [32, 64, 128]
WINDOW_VALUES = [4, 8, 16]
UNITS_BASE = M.LSTM_UNITS     # 64
WINDOW_BASE = P.WINDOW_SIZE   # 8
F1_LSTM_FASE3 = 0.749         # referencia da Fase 3 (30 runs) p/ sanity check
TOLERANCIA_SANITY = 0.05      # divergencia maxima aceitavel

if len(sys.argv) >= 3:
    N_RUNS_HP = int(sys.argv[1])
    EPOCHS = int(sys.argv[2])

COLUNAS_CSV = ["experimento", "valor", "run", "seed", "f1_macro", "tempo_s"]


# --------------------------------------------------------------------------
# Retomada (mesmo padrao da Fase 3): le o que ja foi feito antes de rodar.
# --------------------------------------------------------------------------
def carregar_feitos(caminho_csv):
    feitos = set()
    if not caminho_csv.exists():
        return feitos
    with open(caminho_csv, newline="") as f:
        for linha in csv.DictReader(f):
            feitos.add((linha["experimento"], linha["valor"], int(linha["run"])))
    return feitos


def anexar_linha_csv(caminho_csv, linha):
    novo = not caminho_csv.exists()
    with open(caminho_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUNAS_CSV)
        if novo:
            w.writeheader()
        w.writerow({c: linha[c] for c in COLUNAS_CSV})


def treinar_lstm_run(seed, units, dados, pesos, input_shape):
    """Treina uma LSTM com 'units' unidades e devolve (f1_macro, tempo_s)."""
    tf.keras.backend.clear_session()  # libera o grafo do run anterior
    M.fixar_sementes(seed)
    modelo = M.construir_lstm(input_shape, units=units)
    t0 = time.perf_counter()
    modelo.fit(dados["X_seq_train"], dados["y_train"],
               epochs=EPOCHS, batch_size=M.BATCH_SIZE,
               class_weight=pesos, verbose=0)
    dt = time.perf_counter() - t0
    y_proba = modelo.predict(dados["X_seq_test"], verbose=0)
    y_pred = y_proba.argmax(axis=1)
    f1 = f1_score(dados["y_test"], y_pred, average="macro")
    return f1, dt


def rodar_experimento(nome_exp, valores, montar_dados_por_valor, feitos):
    """Roda N_RUNS_HP sementes para cada valor do parametro 'nome_exp'.
    'montar_dados_por_valor(valor)' devolve (dados, pesos, input_shape, units).
    Pula combinacoes ja gravadas no CSV (retomada)."""
    for valor in valores:
        dados, pesos, input_shape, units = montar_dados_por_valor(valor)
        for run in range(N_RUNS_HP):
            chave = (nome_exp, str(valor), run)
            if chave in feitos:
                print("  [%s=%s] run %d/%d -> ja feito, pulando."
                      % (nome_exp, valor, run + 1, N_RUNS_HP))
                continue
            seed = run
            f1, dt = treinar_lstm_run(seed, units, dados, pesos, input_shape)
            linha = {"experimento": nome_exp, "valor": valor, "run": run,
                     "seed": seed, "f1_macro": f1, "tempo_s": dt}
            anexar_linha_csv(CSV_OUT, linha)
            print("  [%s=%s] run %d/%d seed=%d  f1_macro=%.4f  t=%.1fs"
                  % (nome_exp, valor, run + 1, N_RUNS_HP, seed, f1, dt))


# --------------------------------------------------------------------------
# PARTE A - LSTM_UNITS (usa o dataset.npz da Fase 1, window=8 fixo)
# --------------------------------------------------------------------------
def montar_dados_units(units):
    """Sempre o mesmo dataset (window=8); so muda 'units' na hora de montar
    o modelo (devolvido junto para o treinador usar)."""
    d = np.load(DATASET, allow_pickle=True)
    dados = {k: d[k] for k in ["X_seq_train", "X_seq_test", "y_train", "y_test"]}
    pesos = M.calcular_pesos_classe(dados["y_train"])
    input_shape = dados["X_seq_train"].shape[1:]
    return dados, pesos, input_shape, units


# --------------------------------------------------------------------------
# PARTE B - WINDOW_SIZE (re-gera as janelas com o tratamento de lacuna)
# --------------------------------------------------------------------------
def montar_dados_window(window):
    """Re-gera o dataset para um 'window' diferente, reaplicando o MESMO
    descarte de lacuna temporal da Fase 1 (passo critico - reforco do diretor:
    o gap e verificado do inicio da janela ATE o alvo t+2, nao so dentro da
    janela de entrada). Imprime o checkpoint de janelas validas/descartadas,
    igual ao da Fase 1."""
    global _DF_ANO
    if "_DF_ANO" not in globals():
        print("  Carregando e recortando o CSV bruto (1x, reaproveitado p/ os 3 windows)...")
        df = P.carregar_e_limpar(DATA_CSV)
        _DF_ANO = P.recortar_periodo(df, P.DIAS_RECORTE)

    dados_janelas = P.construir_janelas(_DF_ANO["g"].values, _DF_ANO["ts"].values,
                                        window, P.HORIZON, P.GAP_MAX_MIN)
    n_poss = dados_janelas["n_possiveis"]
    n_desc = dados_janelas["n_descartadas"]
    n_val = len(dados_janelas["y"])
    print("    [window=%d] janelas possiveis=%d | descartadas por gap=%d (%.1f%%) | validas=%d"
          % (window, n_poss, n_desc, 100.0 * n_desc / n_poss, n_val))

    split = P.dividir_e_escalar(dados_janelas, P.FRAC_TREINO)
    dados = {"X_seq_train": split["X_seq_train"], "X_seq_test": split["X_seq_test"],
             "y_train": split["y_train"], "y_test": split["y_test"]}
    pesos = M.calcular_pesos_classe(dados["y_train"])
    input_shape = dados["X_seq_train"].shape[1:]
    return dados, pesos, input_shape, UNITS_BASE


# --------------------------------------------------------------------------
# Graficos e resumo
# --------------------------------------------------------------------------
def imprimir_resumo(df):
    print("\n" + "=" * 78)
    print("RESUMO DA FASE 5 - VARIACAO DE HIPERPARAMETROS")
    print("=" * 78)

    print("\n[A] LSTM_UNITS (dataset fixo, window=8) - F1-macro media +/- desvio:")
    for v in UNITS_VALUES:
        sub = df[(df["experimento"] == "units") & (df["valor"] == str(v))]
        print("    units=%3d  f1_macro=%.4f +/- %.4f  (mediana=%.4f, %d runs)"
              % (v, sub["f1_macro"].mean(), sub["f1_macro"].std(),
                 sub["f1_macro"].median(), len(sub)))

    print("\n[B] WINDOW_SIZE (re-gera janelas) - F1-macro media +/- desvio:")
    for v in WINDOW_VALUES:
        sub = df[(df["experimento"] == "window") & (df["valor"] == str(v))]
        print("    window=%2d  f1_macro=%.4f +/- %.4f  (mediana=%.4f, %d runs)"
              % (v, sub["f1_macro"].mean(), sub["f1_macro"].std(),
                 sub["f1_macro"].median(), len(sub)))

    # Sanity check cruzado: units=64 (Parte A) deve ficar perto do F1=0,749
    # da Fase 3 (30 runs). Window=8 (Parte B) e o mesmo ponto re-derivado dos
    # dados brutos - serve de segunda confirmacao independente.
    sub_base_a = df[(df["experimento"] == "units") & (df["valor"] == str(UNITS_BASE))]
    sub_base_b = df[(df["experimento"] == "window") & (df["valor"] == str(WINDOW_BASE))]
    f1_base_a = sub_base_a["f1_macro"].mean()
    f1_base_b = sub_base_b["f1_macro"].mean()
    diff_a = abs(f1_base_a - F1_LSTM_FASE3)
    diff_b = abs(f1_base_b - F1_LSTM_FASE3)
    print("\n[Sanity check cruzado] ponto base units=%d / window=%d:" % (UNITS_BASE, WINDOW_BASE))
    print("    Fase 3 (30 runs, referencia):       f1_macro=%.4f" % F1_LSTM_FASE3)
    print("    Fase 5 - Parte A (dataset.npz):      f1_macro=%.4f  (diff=%.4f) %s"
          % (f1_base_a, diff_a, "OK" if diff_a <= TOLERANCIA_SANITY else "** DIVERGENTE - INVESTIGAR **"))
    print("    Fase 5 - Parte B (janelas re-geradas): f1_macro=%.4f  (diff=%.4f) %s"
          % (f1_base_b, diff_b, "OK" if diff_b <= TOLERANCIA_SANITY else "** DIVERGENTE - INVESTIGAR **"))

    print("\n[Grafico] a figura F1 vs hiperparametro do conjunto de entrega e")
    print("    gerada pela Fase 6 (06_f1_vs_hiperparametro.png), a partir deste")
    print("    mesmo CSV - evita figura duplicada (decisao do diretor).")
    print("=" * 78 + "\n")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Config: N_RUNS_HP=%d | EPOCHS=%d" % (N_RUNS_HP, EPOCHS))

    feitos = carregar_feitos(CSV_OUT)
    if feitos:
        print("Retomando: %d combinacoes ja concluidas serao puladas." % len(feitos))

    print("\n--- Parte A: LSTM_UNITS %s (dataset fixo, window=%d) ---"
          % (UNITS_VALUES, WINDOW_BASE))
    rodar_experimento("units", UNITS_VALUES, montar_dados_units, feitos)

    print("\n--- Parte B: WINDOW_SIZE %s (re-gera janelas, units=%d) ---"
          % (WINDOW_VALUES, UNITS_BASE))
    rodar_experimento("window", WINDOW_VALUES, montar_dados_window, feitos)

    df = pd.read_csv(CSV_OUT, dtype={"valor": str})
    imprimir_resumo(df)
    print("CSV salvo em: %s" % CSV_OUT)


if __name__ == "__main__":
    main()
