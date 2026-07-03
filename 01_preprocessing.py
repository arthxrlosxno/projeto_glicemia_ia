# -*- coding: utf-8 -*-
"""
FASE 1 - Pre-processamento e Engenharia de Features
====================================================

Objetivo: transformar o CSV bruto do FreeStyle Libre em dois conjuntos prontos
para os modelos da Fase 2:

  - TABULAR   (n, n_features)  -> baseline Random Forest (sklearn)
  - SEQUENCIAL (n, 8, 1)       -> modelos de Deep Learning (LSTM/GRU/CNN)

O alvo (y) e a classe de glicemia 30 min a frente (t+2 leituras):
  classe 0 = hipo   (< 70 mg/dL)
  classe 1 = normal (70 a 180 mg/dL)
  classe 2 = hiper  (> 180 mg/dL)

"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Constantes do problema 
WINDOW_SIZE = 8       # leituras de entrada = 2h de historico
HORIZON = 2           # leituras a frente = 30 min
DIAS_RECORTE = 365    # escopo: ultimos 12 meses
GAP_MAX_MIN = 30      # buraco maior que isso entre leituras = descartar janela
FRAC_TREINO = 0.80    # split cronologico 80/20

# Posicao das colunas relevantes no CSV 
COL_TIMESTAMP = 2
COL_TIPO_REGISTRO = 3
COL_GLICOSE = 4

# Caminhos relativos ao projeto 
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_CSV = PROJECT_DIR / "data" / "ArthurLosano_glucose_6-10-2026.csv"
OUT_DIR = PROJECT_DIR / "results" / "processed"


def carregar_e_limpar(caminho):
    """Carrega o CSV, mantem so leitura de sensor (tipo 0), parseia o timestamp
    e ordena cronologicamente. Retorna um DataFrame com colunas 'ts' e 'g'."""
    df = pd.read_csv(caminho, skiprows=1, low_memory=False)

    # Acessa colunas por posicao para evitar problema com nomes acentuados.
    col_ts = df.columns[COL_TIMESTAMP]
    col_tipo = df.columns[COL_TIPO_REGISTRO]
    col_g = df.columns[COL_GLICOSE]

    # Mantem apenas leituras automaticas do sensor.
    df = df[df[col_tipo] == 0].copy()

    # Timestamp no formato MM-DD-YYYY HH:MM AM/PM.
    df["ts"] = pd.to_datetime(df[col_ts], format="%m-%d-%Y %I:%M %p", errors="coerce")
    df["g"] = pd.to_numeric(df[col_g], errors="coerce")

    # Descarta linhas sem timestamp ou sem valor de glicose e ordena no tempo.
    df = df.dropna(subset=["ts", "g"]).sort_values("ts").reset_index(drop=True)
    return df[["ts", "g"]]


def recortar_periodo(df, dias):
    """Mantem apenas os ultimos 'dias' de dados (escopo do projeto)."""
    limite = df["ts"].max() - pd.Timedelta(days=dias)
    return df[df["ts"] >= limite].reset_index(drop=True)


def rotular_classe(valor):
    """Converte um valor de glicose em classe: 0 hipo, 1 normal, 2 hiper."""
    if valor < 70:
        return 0
    if valor <= 180:
        return 1
    return 2


def construir_janelas(g, ts, window, horizon, gap_max_min):
    """Percorre a serie criando janelas deslizantes.

    Para cada posicao i:
      - entrada  = g[i : i+window]            (as 'window' leituras)
      - alvo     = classe de g[i+window+horizon-1]  (t+2 leituras a frente)
    A janela e descartada se qualquer intervalo entre leituras consecutivas no
    trecho usado (entrada + caminho ate o alvo) for maior que gap_max_min.

    """
    lista_tab = []        # 8 leituras brutas + features resumidas (Random Forest)
    lista_seq = []        # serie multicanal (window, n_canais) para o DL
    lista_y = []          # classe alvo
    lista_ts_atual = []   # timestamp da ultima leitura de entrada (para plots/Fase 7)

    n_descartadas = 0
    eixo_x = np.arange(window)  # 0,1,...,window-1 (usado no slope)
    uns = np.ones(window)       # vetor de 1s para replicar features por-janela

    ultimo_inicio = len(g) - window - horizon  # ultimo i valido
    for i in range(ultimo_inicio + 1):
        idx_alvo = i + window + horizon - 1

        # Verifica buracos no trecho entrada + caminho ate o alvo.
        trecho_ts = ts[i: idx_alvo + 1]
        difs_min = np.diff(trecho_ts).astype("timedelta64[m]").astype(int)
        if (difs_min > gap_max_min).any():
            n_descartadas += 1
            continue

        janela = g[i: i + window]

        # ---- Features derivadas (resumo da janela) ----
        glicose_atual = janela[-1]
        media = janela.mean()
        desvio = janela.std()
        # slope: coeficiente angular da reta ajustada (tendencia da janela).
        slope = np.polyfit(eixo_x, janela, 1)[0]
        # derivada (escalar): variacao media por passo nas ultimas leituras.
        derivada = (janela[-1] - janela[-3]) / 2.0
        # hora do dia em sin/cos para capturar o padrao circadiano
        # (sem isso, 23h e 0h pareceriam distantes; com sin/cos ficam proximas).
        t_atual = pd.Timestamp(ts[i + window - 1])
        hora_dec = t_atual.hour + t_atual.minute / 60.0
        hora_sin = np.sin(2 * np.pi * hora_dec / 24.0)
        hora_cos = np.cos(2 * np.pi * hora_dec / 24.0)

        # ---- Formato TABULAR (15 colunas): 8 brutas + 7 resumidas ----
        linha_tab = list(janela) + [glicose_atual, media, desvio, slope,
                                    derivada, hora_sin, hora_cos]
        lista_tab.append(linha_tab)

        # ---- Formato SEQUENCIAL (window, 7 canais) ----
        # Canais naturalmente por-timestep: glicose bruta e derivada (variacao
        # passo a passo). Os demais (media, desvio, slope, hora_sin, hora_cos)
        # sao por-janela; replicamos o mesmo valor nos 8 timesteps,
        # para a rede receber exatamente a mesma informacao do tabular.
        derivada_ts = np.diff(janela, prepend=janela[0])  # variacao por passo (1o = 0)
        canais = np.column_stack([
            janela,            # 0: glicose bruta (por-timestep)
            derivada_ts,       # 1: derivada por-timestep
            media * uns,       # 2: media (replicada)
            desvio * uns,      # 3: desvio (replicado)
            slope * uns,       # 4: slope (replicado)
            hora_sin * uns,    # 5: hora_sin (replicada)
            hora_cos * uns,    # 6: hora_cos (replicada)
        ])
        lista_seq.append(canais)

        lista_y.append(rotular_classe(g[idx_alvo]))
        lista_ts_atual.append(ts[i + window - 1])

    # Nomes das 15 colunas tabulares (8 brutas + 7 resumidas).
    nomes_brutas = ["leitura_%d" % k for k in range(window)]
    nomes_resumidas = ["glicose_atual", "media", "desvio", "slope",
                       "derivada", "hora_sin", "hora_cos"]
    feature_names = nomes_brutas + nomes_resumidas

    # Nomes dos 7 canais sequenciais.
    channel_names = ["glicose_bruta", "derivada_ts", "media", "desvio",
                     "slope", "hora_sin", "hora_cos"]

    return {
        "X_tab": np.array(lista_tab, dtype=float),
        "X_seq": np.array(lista_seq, dtype=float),
        "y": np.array(lista_y, dtype=int),
        "ts_atual": np.array(lista_ts_atual),
        "feature_names": feature_names,
        "channel_names": channel_names,
        "n_descartadas": n_descartadas,
        "n_possiveis": ultimo_inicio + 1,
    }


def dividir_e_escalar(dados, frac_treino):
    """Split cronologico 80/20 e normalizacao (StandardScaler) ajustada so no
    treino. Mantem tambem a versao NAO escalada do tabular (necessaria na Fase 7
    para segmentar por desvio/slope em unidades reais)."""
    X_tab = dados["X_tab"]
    X_seq = dados["X_seq"]
    y = dados["y"]

    n = len(y)
    corte = int(n * frac_treino)

    X_tab_tr_raw, X_tab_te_raw = X_tab[:corte], X_tab[corte:]
    X_seq_tr_raw, X_seq_te_raw = X_seq[:corte], X_seq[corte:]
    y_tr, y_te = y[:corte], y[corte:]
    ts_te = dados["ts_atual"][corte:]

    # Scaler do tabular: ajusta so no treino, aplica nos dois.
    scaler_tab = StandardScaler().fit(X_tab_tr_raw)
    X_tab_tr = scaler_tab.transform(X_tab_tr_raw)
    X_tab_te = scaler_tab.transform(X_tab_te_raw)

    # Scaler da serie multicanal: cada canal tem unidade/escala diferente, entao
    # achatamos para (n*window, n_canais) e o StandardScaler normaliza POR CANAL.
    # Fit so no treino, igual ao tabular.
    forma_tr = X_seq_tr_raw.shape
    forma_te = X_seq_te_raw.shape
    n_canais = forma_tr[2]
    scaler_seq = StandardScaler().fit(X_seq_tr_raw.reshape(-1, n_canais))
    X_seq_tr = scaler_seq.transform(X_seq_tr_raw.reshape(-1, n_canais)).reshape(forma_tr)
    X_seq_te = scaler_seq.transform(X_seq_te_raw.reshape(-1, n_canais)).reshape(forma_te)

    return {
        "corte": corte,
        "X_tab_train": X_tab_tr, "X_tab_test": X_tab_te,
        "X_tab_raw_train": X_tab_tr_raw, "X_tab_raw_test": X_tab_te_raw,
        "X_seq_train": X_seq_tr, "X_seq_test": X_seq_te,
        "y_train": y_tr, "y_test": y_te,
        "ts_test": ts_te,
        "scaler_tab_mean": scaler_tab.mean_,
    }


def _distribuicao(y):
    """Retorna (contagens, percentuais) das 3 classes em y."""
    total = len(y)
    cont = [int((y == k).sum()) for k in (0, 1, 2)]
    perc = [100.0 * c / total for c in cont]
    return cont, perc


def imprimir_resumo(df_ano, dados, split):
    """Imprime o RESUMO DA FASE"""
    print("\n" + "=" * 70)
    print("RESUMO DA FASE 1 - PRE-PROCESSAMENTO")
    print("=" * 70)

    # 1) Leituras e estatisticas da glicose
    g = df_ano["g"]
    print("\n[1] Leituras (tipo 0, ultimos 12 meses): %d" % len(df_ano))
    print("    periodo: %s -> %s" % (df_ano["ts"].min().date(), df_ano["ts"].max().date()))
    print("    glicose: media=%.1f  desvio=%.1f  min=%.0f  max=%.0f"
          % (g.mean(), g.std(), g.min(), g.max()))

    # 2) Janelas descartadas por lacuna 
    n_poss = dados["n_possiveis"]
    n_desc = dados["n_descartadas"]
    n_val = len(dados["y"])
    print("\n[2] Janelas possiveis: %d  |  descartadas por gap: %d (%.1f%%)  |  validas: %d"
          % (n_poss, n_desc, 100.0 * n_desc / n_poss, n_val))

    # 3) Distribuicao do alvo
    cont_tot, perc_tot = _distribuicao(dados["y"])
    cont_tr, perc_tr = _distribuicao(split["y_train"])
    cont_te, perc_te = _distribuicao(split["y_test"])
    print("\n[3] Distribuicao do alvo (hipo / normal / hiper):")
    print("    total : %5d (%.1f%%) | %5d (%.1f%%) | %5d (%.1f%%)"
          % (cont_tot[0], perc_tot[0], cont_tot[1], perc_tot[1], cont_tot[2], perc_tot[2]))
    print("    treino: %5d (%.1f%%) | %5d (%.1f%%) | %5d (%.1f%%)"
          % (cont_tr[0], perc_tr[0], cont_tr[1], perc_tr[1], cont_tr[2], perc_tr[2]))
    print("    teste : %5d (%.1f%%) | %5d (%.1f%%) | %5d (%.1f%%)"
          % (cont_te[0], perc_te[0], cont_te[1], perc_te[1], cont_te[2], perc_te[2]))
    print("    -> hipo no teste em torno de ~2,1% e ESPERADO (mais hipos nos meses recentes).")

    # 4) Shapes finais (tabular e sequencial com o mesmo n) 
    print("\n[4] Shapes finais (mesma informacao em dois formatos):")
    print("    TABULAR (RF)  X_tab_train: %s | X_tab_test: %s"
          % (split["X_tab_train"].shape, split["X_tab_test"].shape))
    print("    SEQUENC (DL)  X_seq_train: %s | X_seq_test: %s"
          % (split["X_seq_train"].shape, split["X_seq_test"].shape))
    n_tab = split["X_tab_train"].shape[0] + split["X_tab_test"].shape[0]
    n_seq = split["X_seq_train"].shape[0] + split["X_seq_test"].shape[0]
    print("    n total tabular = %d | n total sequencial = %d | iguais=%s"
          % (n_tab, n_seq, n_tab == n_seq))
    print("    15 colunas tabulares: %s" % (", ".join(dados["feature_names"])))
    print("    7 canais sequenciais: %s" % (", ".join(dados["channel_names"])))

    # 5) Confirmacao do split cronologico (ultimo treino < primeiro teste)
    ts_atual = dados["ts_atual"]
    corte = split["corte"]
    ultimo_treino = pd.Timestamp(ts_atual[corte - 1])
    primeiro_teste = pd.Timestamp(ts_atual[corte])
    ok_crono = ultimo_treino < primeiro_teste
    print("\n[5] Split cronologico: ultimo treino=%s | primeiro teste=%s | OK=%s"
          % (ultimo_treino, primeiro_teste, ok_crono))

    # 6) Confirmacao da normalizacao (treino com media ~0 apos o scaler)
    media_tab_treino = split["X_tab_train"].mean(axis=0)
    print("\n[6] Normalizacao (StandardScaler ajustado so no treino):")
    print("    media de cada feature no treino apos escalar (deve ser ~0):")
    print("    %s" % np.round(media_tab_treino, 3))
    print("=" * 70 + "\n")


def main():
    print("Carregando e limpando o CSV...")
    df = carregar_e_limpar(DATA_CSV)

    print("Recortando os ultimos %d dias..." % DIAS_RECORTE)
    df_ano = recortar_periodo(df, DIAS_RECORTE)

    print("Construindo janelas deslizantes (window=%d, horizon=%d)..." % (WINDOW_SIZE, HORIZON))
    dados = construir_janelas(df_ano["g"].values, df_ano["ts"].values,
                              WINDOW_SIZE, HORIZON, GAP_MAX_MIN)

    pct_treino = round(FRAC_TREINO * 100)
    print("Dividindo (cronologico %d/%d) e normalizando..."
          % (pct_treino, 100 - pct_treino))
    split = dividir_e_escalar(dados, FRAC_TREINO)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    saida = OUT_DIR / "dataset.npz"
    np.savez(
        saida,
        X_tab_train=split["X_tab_train"], X_tab_test=split["X_tab_test"],
        X_tab_raw_train=split["X_tab_raw_train"], X_tab_raw_test=split["X_tab_raw_test"],
        X_seq_train=split["X_seq_train"], X_seq_test=split["X_seq_test"],
        y_train=split["y_train"], y_test=split["y_test"],
        ts_test=split["ts_test"].astype("datetime64[m]"),
        feature_names=np.array(dados["feature_names"]),
        channel_names=np.array(dados["channel_names"]),
    )

    imprimir_resumo(df_ano, dados, split)
    print("Arquivo salvo em: %s" % saida)


if __name__ == "__main__":
    main()
