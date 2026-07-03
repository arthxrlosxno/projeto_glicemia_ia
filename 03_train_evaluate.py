# -*- coding: utf-8 -*-
"""
FASE 3 - Treino, Avaliacao e Metricas (30 runs por modelo)
==========================================================

Para cada um dos 3 modelos (Random Forest, LSTM, GRU) roda N_RUNS execucoes,
cada uma com uma semente diferente. Por run armazena:
  - F1-macro (metrica principal; trata as 3 classes igualmente)
  - AUC one-vs-rest (complementar; com try/except por seguranca)
  - accuracy
  - recall da HIPO (classe 0) - a metrica de maior valor clinico
  - tempo de treino (segundos)

Ao final: media, desvio-padrao e mediana de cada metrica,
matriz de confusao do melhor run de cada modelo, e salva as predicoes do
melhor run

"""

import csv
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             recall_score, roc_auc_score)

import tensorflow as tf

# --------------------------------------------------------------------------
# Caminhos e import do modulo de modelos (02_models.py nao e importavel pelo
# nome por comecar com numero, entao carregamos pelo caminho do arquivo).
# --------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATASET = PROJECT_DIR / "results" / "processed" / "dataset.npz"
RESULTS_DIR = PROJECT_DIR / "results"
PRED_OUT = RESULTS_DIR / "processed" / "predictions_best.npz"
CSV_OUT = RESULTS_DIR / "metrics_30runs.csv"


def carregar_modulo_modelos():
    spec = importlib.util.spec_from_file_location("modelos", SCRIPT_DIR / "02_models.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = carregar_modulo_modelos()

# Permite sobrescrever N_RUNS e EPOCHS pela linha de comando (validacao rapida).
N_RUNS = 30
EPOCHS = M.EPOCHS
if len(sys.argv) >= 3:
    N_RUNS = int(sys.argv[1])
    EPOCHS = int(sys.argv[2])


def calcular_metricas(y_true, y_pred, y_proba):
    """Calcula as metricas de um run. A AUC vai num try/except: se em algum
    recorte faltar uma classe, devolve NaN em vez de quebrar o pipeline."""
    f1 = f1_score(y_true, y_pred, average="macro")
    acc = accuracy_score(y_true, y_pred)
    # recall por classe; pegamos o da hipo (classe 0).
    recall_hipo = recall_score(y_true, y_pred, labels=[0, 1, 2], average=None,
                               zero_division=0)[0]
    try:
        auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")
    return {"f1_macro": f1, "auc": auc, "accuracy": acc, "recall_hipo": recall_hipo}


def treinar_random_forest(seed, dados):
    """Treina o RF e retorna predicoes, probabilidades e tempo."""
    rf = M.construir_random_forest(seed=seed)
    t0 = time.perf_counter()
    rf.fit(dados["X_tab_train"], dados["y_train"])
    dt = time.perf_counter() - t0
    y_pred = rf.predict(dados["X_tab_test"])
    y_proba = rf.predict_proba(dados["X_tab_test"])
    return y_pred, y_proba, dt


def treinar_keras(construtor, seed, dados, pesos, input_shape):
    """Treina um modelo Keras (LSTM ou GRU) e retorna predicoes/probas/tempo."""
    tf.keras.backend.clear_session()  # libera o grafo do run anterior
    M.fixar_sementes(seed)
    modelo = construtor(input_shape)
    t0 = time.perf_counter()
    modelo.fit(dados["X_seq_train"], dados["y_train"],
               epochs=EPOCHS, batch_size=M.BATCH_SIZE,
               class_weight=pesos, verbose=0)
    dt = time.perf_counter() - t0
    y_proba = modelo.predict(dados["X_seq_test"], verbose=0)
    y_pred = y_proba.argmax(axis=1)
    return y_pred, y_proba, dt


# Ordem das colunas do CSV e prefixo das chaves de predicoes por modelo.
COLUNAS_CSV = ["modelo", "run", "seed", "f1_macro", "auc", "accuracy", "recall_hipo", "tempo_s"]
PREFIXO = {"RandomForest": "rf", "LSTM": "lstm", "GRU": "gru"}


def carregar_feitos(caminho_csv):
    """Le o CSV (se existir) e devolve o conjunto de pares (modelo, run) ja
    concluidos. E o que permite RETOMAR de onde parou apos uma queda."""
    feitos = set()
    if not caminho_csv.exists():
        return feitos
    with open(caminho_csv, newline="") as f:
        for linha in csv.DictReader(f):
            feitos.add((linha["modelo"], int(linha["run"])))
    return feitos


def anexar_linha_csv(caminho_csv, met):
    """Grava UMA linha de metricas no CSV imediatamente (append). Escreve o
    cabecalho se o arquivo ainda nao existe."""
    novo = not caminho_csv.exists()
    with open(caminho_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUNAS_CSV)
        if novo:
            w.writeheader()
        w.writerow({c: met[c] for c in COLUNAS_CSV})


def carregar_preds(caminho_npz, y_test):
    """Carrega as predicoes ja salvas (se houver) num dicionario; senao comeca
    so com o y_test."""
    preds = {"y_test": y_test}
    if caminho_npz.exists():
        d = np.load(caminho_npz, allow_pickle=True)
        for chave in d.files:
            preds[chave] = d[chave]
    return preds


def salvar_preds(caminho_npz, preds):
    """Regrava o arquivo de predicoes do melhor run (sobrescreve)."""
    caminho_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(caminho_npz, **preds)


def melhor_f1_por_modelo(caminho_csv):
    """Le o CSV e devolve {modelo: melhor_f1_ja_visto} (para retomar o controle
    do melhor run sem perder o que ja tinha sido salvo)."""
    melhor = {}
    if not caminho_csv.exists():
        return melhor
    with open(caminho_csv, newline="") as f:
        for linha in csv.DictReader(f):
            f1 = float(linha["f1_macro"])
            nome = linha["modelo"]
            if f1 > melhor.get(nome, -1.0):
                melhor[nome] = f1
    return melhor


def rodar_modelo(nome, tipo, dados, pesos, input_shape, feitos, preds, melhor_f1):
    """Roda as N_RUNS execucoes de um modelo, pulando as que ja constam no CSV.
    Grava cada run no CSV assim que termina e atualiza/salva as predicoes do
    melhor run de forma incremental (a prova de queda)."""
    prefixo = PREFIXO[nome]
    for run in range(N_RUNS):
        if (nome, run) in feitos:
            print("  [%s] run %2d/%d -> ja feito, pulando." % (nome, run + 1, N_RUNS))
            continue

        seed = run  # semente diferente por run
        if tipo == "rf":
            y_pred, y_proba, dt = treinar_random_forest(seed, dados)
        elif tipo == "lstm":
            y_pred, y_proba, dt = treinar_keras(M.construir_lstm, seed, dados, pesos, input_shape)
        else:  # gru
            y_pred, y_proba, dt = treinar_keras(M.construir_gru, seed, dados, pesos, input_shape)

        met = calcular_metricas(dados["y_test"], y_pred, y_proba)
        met.update({"modelo": nome, "run": run, "seed": seed, "tempo_s": dt})

        # Grava a linha JA (resume-safe).
        anexar_linha_csv(CSV_OUT, met)

        print("  [%s] run %2d/%d seed=%d  f1=%.3f  recall_hipo=%.3f  auc=%.3f  acc=%.3f  t=%.1fs"
              % (nome, run + 1, N_RUNS, seed, met["f1_macro"], met["recall_hipo"],
                 met["auc"], met["accuracy"], dt))

        # Atualiza o melhor run deste modelo e salva as predicoes na hora.
        if met["f1_macro"] > melhor_f1.get(nome, -1.0):
            melhor_f1[nome] = met["f1_macro"]
            preds[prefixo + "_pred"] = y_pred
            preds[prefixo + "_proba"] = y_proba
            salvar_preds(PRED_OUT, preds)


def garantir_predicoes(df, dados, pesos, input_shape, preds):
    """Rede de seguranca: se as predicoes de algum modelo faltarem no arquivo
    (ex.: a melhor run aconteceu antes de uma queda e nao foi salva), re-treina
    a melhor semente daquele modelo para regenerar as predicoes."""
    for nome, tipo in [("RandomForest", "rf"), ("LSTM", "lstm"), ("GRU", "gru")]:
        prefixo = PREFIXO[nome]
        if (prefixo + "_pred") in preds:
            continue
        sub = df[df["modelo"] == nome]
        seed = int(sub.loc[sub["f1_macro"].idxmax(), "seed"])
        print("  regenerando predicoes do melhor run de %s (seed=%d)..." % (nome, seed))
        if tipo == "rf":
            y_pred, y_proba, _ = treinar_random_forest(seed, dados)
        elif tipo == "lstm":
            y_pred, y_proba, _ = treinar_keras(M.construir_lstm, seed, dados, pesos, input_shape)
        else:
            y_pred, y_proba, _ = treinar_keras(M.construir_gru, seed, dados, pesos, input_shape)
        preds[prefixo + "_pred"] = y_pred
        preds[prefixo + "_proba"] = y_proba
    salvar_preds(PRED_OUT, preds)


def imprimir_resumo(df, preds, y_test):
    print("\n" + "=" * 78)
    print("RESUMO DA FASE 3 - %d RUNS POR MODELO" % N_RUNS)
    print("=" * 78)

    metricas = ["f1_macro", "auc", "accuracy", "recall_hipo", "tempo_s"]
    print("\n[Tabela: media +/- desvio (mediana) por modelo]")
    for nome in df["modelo"].unique():
        sub = df[df["modelo"] == nome]
        print("\n  %s" % nome)
        for m in metricas:
            print("    %-12s media=%.4f  desvio=%.4f  mediana=%.4f"
                  % (m, sub[m].mean(), sub[m].std(), sub[m].median()))

    print("\n[Matriz de confusao do melhor run de cada modelo] (linhas=real, colunas=previsto)")
    print("  classes: 0=hipo 1=normal 2=hiper")
    for nome in df["modelo"].unique():
        sub = df[df["modelo"] == nome]
        melhor_run = int(sub.loc[sub["f1_macro"].idxmax(), "run"])
        y_pred = preds[PREFIXO[nome] + "_pred"]
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
        print("\n  %s (melhor run=%d):" % (nome, melhor_run))
        for i, linha in enumerate(cm):
            print("    real %d -> %s" % (i, linha.tolist()))

    print("\n[Checkpoints Secao 9b]")
    f1_trivial = 0.22
    for nome in df["modelo"].unique():
        sub = df[df["modelo"] == nome]
        f1m = sub["f1_macro"].mean()
        rec = sub["recall_hipo"].mean()
        var_ok = sub["f1_macro"].std() > 1e-9
        print("  %-13s F1-macro medio=%.3f (>%.2f trivial=%s) | recall_hipo medio=%.3f (>0=%s) | variacao entre runs=%s"
              % (nome, f1m, f1_trivial, f1m > f1_trivial, rec, rec > 0, var_ok))
    print("  Linhas no CSV: %d (esperado %d = %d modelos x %d runs)"
          % (len(df), 3 * N_RUNS, 3, N_RUNS))
    print("=" * 78 + "\n")


def main():
    print("Carregando dataset da Fase 1...")
    d = np.load(DATASET, allow_pickle=True)
    dados = {k: d[k] for k in ["X_tab_train", "X_tab_test", "X_seq_train",
                               "X_seq_test", "y_train", "y_test"]}
    y_test = dados["y_test"]
    input_shape = dados["X_seq_train"].shape[1:]
    pesos = M.calcular_pesos_classe(dados["y_train"])

    print("Config: N_RUNS=%d | EPOCHS=%d | input_shape=%s" % (N_RUNS, EPOCHS, input_shape))
    print("Pesos de classe: hipo=%.2f normal=%.2f hiper=%.2f" % (pesos[0], pesos[1], pesos[2]))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # RETOMADA: descobre o que ja foi feito e carrega as predicoes ja salvas.
    feitos = carregar_feitos(CSV_OUT)
    melhor_f1 = melhor_f1_por_modelo(CSV_OUT)
    preds = carregar_preds(PRED_OUT, y_test)
    if feitos:
        print("Retomando: %d runs ja concluidos serao pulados." % len(feitos))
    print("")

    for nome, tipo in [("RandomForest", "rf"), ("LSTM", "lstm"), ("GRU", "gru")]:
        print("Treinando %s (%d runs)..." % (nome, N_RUNS))
        rodar_modelo(nome, tipo, dados, pesos, input_shape, feitos, preds, melhor_f1)

    # Le o CSV completo (inclui runs retomados) para o resumo final.
    df = pd.read_csv(CSV_OUT)
    garantir_predicoes(df, dados, pesos, input_shape, preds)

    imprimir_resumo(df, preds, y_test)
    print("CSV salvo em:        %s" % CSV_OUT)
    print("Predicoes salvas em: %s" % PRED_OUT)


if __name__ == "__main__":
    main()
