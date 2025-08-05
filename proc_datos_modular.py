import pandas as pd
import numpy as np
import torch


## Eleccion datasets ##
datasets = ["datasets/nba_pergame_24y25.csv",
            "datasets/nba_pergame_25_full.csv",
            "datasets/nba_pergame_24_full.csv",
            "datasets/roster_hawks_pergame_25.csv",
            "datasets/roster_celtics_pergame_25.csv",
            "datasets/roster_knicks_pergame_25.csv",
            "datasets/roster_lakers_pergame_24.csv",
            "datasets/roster_clippers_pergame_24.csv",
            "datasets/roster_warrior_pergame_24.csv",
            "datasets/roster_kings_pergame_24.csv",
            "datasets/roster_thunder_pergame_24.csv"]

nombre_set_entrenamiento = "datasets/nba_pergame_24_full.csv"
nombre_set_test = "datasets/roster_hawks_pergame_25.csv"

## PARAMETROS a ajustar ##

hacer_prints = False

modo_autoencoder = True # El autoencoder usara entrada=salida

hay_fila_totales_entrenamiento = False
hay_fila_totales_test = True

quitar_ruido_test = False

modos_posibles =["completo", "solo_volumen", "eficiencia_pura", "reducido"]
modo_columnas = "solo_volumen"

umbral_partidos = 10 # min partidos para filtrar ruido
umbral_minutos = 10  # min minutos para filtrar ruido

normalizar_datos = True
modos_norm_posibles = ["zscore", "minmax"]
modo_normalizacion = "zscore"

modos_etiquetado = ["posicion"]
modo_etiquetado = "posicion"

def procesar_datos(
    nombre_set_entrenamiento,
    nombre_set_test,
    modo_columnas="solo_volumen",
    modo_etiquetado = None,
    modo_normalizacion="zscore",
    normalizar_datos=True,
    modo_autoencoder=False,
    hacer_prints=False,
    umbral_partidos=10,
    umbral_minutos=10,
    hay_fila_totales_entrenamiento=False,
    hay_fila_totales_test=True,
    quitar_ruido_test=False
):
    #----------------------------- AVISOS -----------------------------
    if modo_autoencoder:
        print("\nAVISO: se estan procesando datos para entrenar un autoencoder.\n")
    else:
        print("\nAVISO: se estan procesando datos para entrenar un MLP.\n")

    # ------------------- CONFIGURACIÓN DE COLUMNAS -------------------
    column_sets = {
        "completo": [
            "G", "GS", "MP", "FG", "FG%", "3P", "3P%", "2P", "2P%", "eFG%",
            "FT", "FT%", "ORB", "DRB", "AST", "STL", "BLK", "TOV", "PF"
        ],
        "solo_volumen": [
            "G", "GS", "MP", "FG", "FGA", "3P", "3PA", "2P", "2PA",
            "FT", "FTA", "ORB", "DRB", "AST", "STL", "BLK", "TOV", "PF"
        ],
        "eficiencia_pura": ["FG%", "3P%", "2P%", "eFG%", "FT%"],
        "reducido": ["MP", "FG", "3P", "FT", "AST", "REB", "TOV"]
    }

    cols_a_usar = column_sets[modo_columnas]

    # ------------------- LECTURA Y FILTRADO INICIAL -------------------
    def filtrar_fila_total(df):
        resultado = []
        for nombre, grupo in df.groupby("Player"):
            total_row = grupo[grupo["Team"] == "2TM"]
            if not total_row.empty:
                resultado.append(total_row)
            else:
                resultado.append(grupo.iloc[:1])
        return pd.concat(resultado)

    df_train = pd.read_csv(nombre_set_entrenamiento, encoding="utf-8")
    df_train = filtrar_fila_total(df_train)

    df_test = pd.read_csv(nombre_set_test, encoding="utf-8")

    df_train = df_train[(df_train["G"] >= umbral_partidos) & (df_train["MP"] >= umbral_minutos)]
    if quitar_ruido_test:
        df_test = df_test[(df_test["G"] >= umbral_partidos) & (df_test["MP"] >= umbral_minutos)]

    # ------------------- SELECCIÓN DE COLUMNAS Y DROPNA -------------------
    df_train = df_train[cols_a_usar].dropna()
    df_test = df_test[cols_a_usar].dropna()

    if hay_fila_totales_entrenamiento:
        df_train = df_train[:-1]
    if hay_fila_totales_test:
        df_test = df_test[:-1]

    X_train_raw = df_train.to_numpy()
    X_test_raw = df_test.to_numpy()
    X_total_raw = pd.read_csv(nombre_set_entrenamiento, encoding="utf-8")
    X_total_raw = filtrar_fila_total(X_total_raw)[cols_a_usar].dropna().to_numpy()

    # ------------------- NORMALIZACIÓN -------------------
    if normalizar_datos:
        if modo_normalizacion == "zscore":
            mu = X_total_raw.mean(axis=0)
            sigma = X_total_raw.std(axis=0)
            sigma[sigma == 0] = 1e-8

            umbral_std = 1e-4
            columnas_validas = sigma > umbral_std

            if hacer_prints:
                eliminadas = [col for col, keep in zip(cols_a_usar, columnas_validas) if not keep]
                print(f"[INFO] Columnas eliminadas por varianza baja: {eliminadas}")

            mu = mu[columnas_validas]
            sigma = sigma[columnas_validas]
            X_train = (X_train_raw[:, columnas_validas] - mu) / sigma
            X_test = (X_test_raw[:, columnas_validas] - mu) / sigma

        elif modo_normalizacion == "minmax":
            min_vals = X_total_raw.min(axis=0)
            max_vals = X_total_raw.max(axis=0)
            rango = max_vals - min_vals
            rango[rango == 0] = 1e-8

            X_train = (X_train_raw - min_vals) / rango
            X_test = (X_test_raw - min_vals) / rango
    else:
        X_train = X_train_raw
        X_test = X_test_raw

    # ------------------- TARGET -------------------
    if not modo_autoencoder:
        pos_train = pd.read_csv(nombre_set_entrenamiento)["Pos"].tolist()
        pos_test = pd.read_csv(nombre_set_test)["Pos"].tolist()
        if hay_fila_totales_entrenamiento:
            pos_train = pos_train[:-1]
        if hay_fila_totales_test:
            pos_test = pos_test[:-1]

        mapping_pos = {
            "PG": [1,0,0,0,0], "SG": [0,1,0,0,0],
            "SF": [0,0,1,0,0], "PF": [0,0,0,1,0],
            "C":  [0,0,0,0,1]
        }

        y_train = [mapping_pos.get(pos) for pos in pos_train]
        y_test = [mapping_pos.get(pos) for pos in pos_test]

        # Limpieza de None
        cleaned_train = [(x, y) for x, y in zip(X_train, y_train) if y is not None]
        X_train, y_train = zip(*cleaned_train)
        cleaned_test = [(x, y) for x, y in zip(X_test, y_test) if y is not None]
        X_test, y_test = zip(*cleaned_test)
    else:
        y_train = X_train
        y_test = X_test

    # ------------------- CONVERSIÓN A TENSORES -------------------
    X_train_tensor = torch.tensor(np.array(X_train), dtype=torch.float32)
    Y_train_tensor = torch.tensor(np.array(y_train), dtype=torch.float32)
    X_test_tensor  = torch.tensor(np.array(X_test),  dtype=torch.float32)
    Y_test_tensor  = torch.tensor(np.array(y_test),  dtype=torch.float32)

    # ------------------- CHECK FINAL -------------------
    if torch.isnan(X_train_tensor).any():
        print("[ERROR] X_train_tensor contiene NaNs")
    
    # ------------------- ETIQUETADO ---------------------
    if modo_etiquetado is not None:
        if modo_etiquetado == "posicion":
            etiquetas_train = pd.read_csv(nombre_set_entrenamiento)["Pos"].tolist()
            etiquetas_test = pd.read_csv(nombre_set_test)["Pos"].tolist()
            if hay_fila_totales_entrenamiento:
                etiquetas_train = etiquetas_train[:-1]
            if hay_fila_totales_test:
                etiquetas_test = etiquetas_test[:-1]
        else:
            etiquetas_train = None
            etiquetas_test = None
    else:
        etiquetas_train = None
        etiquetas_test = None

    return X_train_tensor, Y_train_tensor, X_test_tensor, Y_test_tensor, etiquetas_train, etiquetas_test


Xs_entrenamiento_def, Ys_entrenamiento_def, Xs_test_def, Ys_test_def, etiquetas_entrenamiento, etiquetas_test = procesar_datos(     nombre_set_entrenamiento,
                                                                                                                                    nombre_set_test,
                                                                                                                                    modo_columnas,
                                                                                                                                    modo_etiquetado,
                                                                                                                                    modo_normalizacion,
                                                                                                                                    normalizar_datos,
                                                                                                                                    modo_autoencoder,
                                                                                                                                    hacer_prints,
                                                                                                                                    umbral_partidos,
                                                                                                                                    umbral_minutos,
                                                                                                                                    hay_fila_totales_entrenamiento,
                                                                                                                                    hay_fila_totales_test,
                                                                                                                                    quitar_ruido_test
                                                                                                                                )



"""
POSIBLES MEJORAS: 
-> AÑADIR ETIQUETADO (PARA ENTRENAMIENTO Y/O VISUALIZACION) YA SEA POR POSICIONES O OTRO TIPO DE CLASIFICACIO
-> PUEDEN SER INTERESANTES LOS DATOS DE ALTURA Y WINGSPAN (DICEN MUCHO DE UN JUGADOR)

"""