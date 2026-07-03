# -*- coding: utf-8 -*-
"""
FASE 2 - Definicao dos Tres Modelos
===================================

Tres modelos de complexidade crescente, todos recebendo a MESMA informacao-base
, em formatos diferentes:

  1. Random Forest (sklearn)   -> entrada TABULAR 
  2. LSTM          (Keras)     -> entrada SEQUENC.
  3. GRU           (Keras)     -> entrada SEQUENC.

"""

import os
import random
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, LSTM, GRU, Dropout, Dense


# Hiperparametros base (mesmos do documento de contexto)

LSTM_UNITS = 64
DROPOUT = 0.2
EPOCHS = 30     
BATCH_SIZE = 32
N_ARVORES = 200
N_CLASSES = 3

EPOCHS_SMOKE = 3 

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATASET = PROJECT_DIR / "results" / "processed" / "dataset.npz"


def fixar_sementes(seed):
    """Fixa as sementes para reprodutibilidade (numpy, random e tensorflow)."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def calcular_pesos_classe(y_train):
    """Calcula pesos inversamente proporcionais a frequencia de cada classe.

    Como a hipo (classe 0) e rara, sem peso o modelo tende a ignora-la. O peso
    'balanced' faz cada classe contribuir igualmente para a loss.
    """
    classes = np.array([0, 1, 2])
    pesos = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return {int(c): float(p) for c, p in zip(classes, pesos)}


def construir_random_forest(seed):
    """Baseline interpretavel. class_weight='balanced' trata o desbalanceamento."""
    return RandomForestClassifier(
        n_estimators=N_ARVORES,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )


def construir_lstm(input_shape, units=LSTM_UNITS, dropout=DROPOUT):
    """LSTM simples: uma camada recorrente + dropout + saida softmax de 3 classes."""
    modelo = Sequential([
        Input(shape=input_shape),
        LSTM(units),
        Dropout(dropout),
        Dense(N_CLASSES, activation="softmax"),
    ], name="LSTM")
    modelo.compile(optimizer="adam",
                   loss="sparse_categorical_crossentropy",
                   metrics=["accuracy"])
    return modelo


def construir_gru(input_shape, units=LSTM_UNITS, dropout=DROPOUT):
    """GRU: mesma estrutura da LSTM, trocando a celula recorrente. Serve para
    comparar arquiteturas recorrentes com a mesma entrada."""
    modelo = Sequential([
        Input(shape=input_shape),
        GRU(units),
        Dropout(dropout),
        Dense(N_CLASSES, activation="softmax"),
    ], name="GRU")
    modelo.compile(optimizer="adam",
                   loss="sparse_categorical_crossentropy",
                   metrics=["accuracy"])
    return modelo


def carregar_dados():
    """Carrega os arrays prontos """
    d = np.load(DATASET, allow_pickle=True)
    return d


def smoke_test():
    """Confere que os 3 modelos montam e treinam sem erro de shape, e que a loss
    desce. Nao e o treino real ,so poucas epocas."""
    print("Carregando dataset da Fase 1...")
    d = carregar_dados()
    X_tab_tr, y_tr = d["X_tab_train"], d["y_train"]
    X_seq_tr = d["X_seq_train"]
    input_shape = X_seq_tr.shape[1:]  # (8, 7)

    fixar_sementes(42)

    # ---- Pesos de classe (confirmar que o balanceamento sera aplicado) ----
    pesos = calcular_pesos_classe(y_tr)
    print("\n" + "=" * 70)
    print("SMOKE TEST - FASE 2 (definicao dos modelos)")
    print("=" * 70)
    print("\n[pesos de classe 'balanced' (Keras class_weight)]")
    print("    hipo(0)=%.2f  normal(1)=%.2f  hiper(2)=%.2f"
          % (pesos[0], pesos[1], pesos[2]))
    print("    (hipo deve ter o maior peso, por ser a classe mais rara)")

    # ---- Random Forest ----
    print("\n[1] Random Forest (tabular %s) - class_weight='balanced'" % (X_tab_tr.shape,))
    rf = construir_random_forest(seed=42)
    rf.fit(X_tab_tr, y_tr)
    acc_tr = rf.score(X_tab_tr, y_tr)
    print("    treinou OK | accuracy no treino=%.3f | n_features=%d"
          % (acc_tr, rf.n_features_in_))

    # ---- LSTM ----
    print("\n[2] LSTM (sequencial %s) - Dense(3, softmax)" % (input_shape,))
    lstm = construir_lstm(input_shape)
    lstm.summary()
    hist = lstm.fit(X_seq_tr, y_tr, epochs=EPOCHS_SMOKE, batch_size=BATCH_SIZE,
                    class_weight=pesos, verbose=2)
    losses = hist.history["loss"]
    print("    loss por epoca: %s | desceu=%s"
          % (["%.4f" % x for x in losses], losses[-1] < losses[0]))

    # ---- GRU ----
    print("\n[3] GRU (sequencial %s) - Dense(3, softmax)" % (input_shape,))
    gru = construir_gru(input_shape)
    gru.summary()
    hist = gru.fit(X_seq_tr, y_tr, epochs=EPOCHS_SMOKE, batch_size=BATCH_SIZE,
                   class_weight=pesos, verbose=2)
    losses = hist.history["loss"]
    print("    loss por epoca: %s | desceu=%s"
          % (["%.4f" % x for x in losses], losses[-1] < losses[0]))

if __name__ == "__main__":
    smoke_test()
