import pandas as pd
import numpy as np
import torch


## FUNCIONES RELEVANTES ##

    
column_sets = {
# ------------------- CONFIGURACIÓN DE COLUMNAS -------------------
        "completo": [
            "G", "GS", "MP", "FG", "FG%", "3P", "3P%", "2P", "2P%", "eFG%",
            "FT", "FT%", "ORB", "DRB", "AST", "STL", "BLK", "TOV", "PF"
        ],
        "solo_volumen": [
            "G", "GS", "MP", "FG", "FGA", "3P", "3PA", "2P", "2PA",
            "FT", "FTA", "ORB", "DRB", "AST", "STL", "BLK", "TOV", "PF"
        ],
        "eficiencia_pura": ["FG%", "3P%", "2P%", "eFG%", "FT%"
        ],
        "reducido": ["MP", "FG", "3P", "FT", "AST", "REB", "TOV"]
}

mapping_pos_onehot = {
            "PG": [1,0,0,0,0], "SG": [0,1,0,0,0],
            "SF": [0,0,1,0,0], "PF": [0,0,0,1,0],
            "C":  [0,0,0,0,1]
}

def filtrar_repetidos(df):
        resultado = []
        if "Team" in df.columns:
            for nombre, grupo in df.groupby("Player"):
                total_row_2 = grupo[grupo["Team"] == "2TM"]
                total_row_3 = grupo[grupo["Team"] == "3TM"] # Revisar esto no falle
                total_row_4 = grupo[grupo["Team"] == "4TM"] # Revisar esto no falle
                if not total_row_2.empty:
                    resultado.append(total_row_2)
                elif not total_row_3.empty:
                    resultado.append(total_row_3)
                elif not total_row_4.empty:
                    resultado.append(total_row_4)
                else:
                    resultado.append(grupo.iloc[:1])
            return pd.concat(resultado)
        else:
            return df#aquiiiiiiiiii


def procesar_datos( archivo_set_train: str,
                    archivo_set_test: str,
                    modo_autoencoder: bool=False,
                    modo_columnas: str="solo_volumen",
                    modo_targets: str="pos",
                    modo_etiquetado: str=None,
                    normalizar_datos: bool=False,
                    modo_normalizacion: str="zscore",
                    umbral_partidos: int=1,
                    umbral_minutos: int=0,
                    umbral_en_test: bool=True,
                    hay_fila_total_entrenamiento : bool=False,
                    hay_fila_total_test: bool=True):
    
    # --------------- AVISOS --------------- #
    if modo_autoencoder:
        print("\nAVISO: se estan procesando datos para un autoencoder.\n")
    else:
        print("\nAVISO: se estan procesando datos para un MLP.\n")
    
    if normalizar_datos:
        print(f"\nAVISO: se van a normalizar los datos en modo {modo_normalizacion}.\n")
        if modo_normalizacion is None:
            print(f"\nAVISO: No se ha dado un modo de normalización y sera zscore por defecto.\n")
    else:
        print(f"\nAVISO: no se van a normalizar los datos.\n")
    
    if umbral_en_test:
        print(f"\nAVISO: se han filtrado los datos de test para jugadores con partidos o minutos insuficientes .\n")
    # --------------- DATOS POR DEFECTO --------------- #


    # --------------- LECTURA INICIAL --------------- #
    df_train = pd.read_csv(archivo_set_train, encoding="utf-8")
    df_test  = pd.read_csv(archivo_set_test, encoding="utf-8")
    
    # --------------- FILTRO REPETIDOS --------------- #
    df_train_sin_repes = filtrar_repetidos(df_train)
    df_test_sin_repes  = filtrar_repetidos(df_test)
    
    # --------------- FILTRO UMBRAL --------------- #
    df_train_umbral = df_train_sin_repes[(df_train_sin_repes["G"] >= umbral_partidos) & (df_train_sin_repes["MP"] >= umbral_minutos)]
    if umbral_en_test:
        df_test_umbral  = df_test_sin_repes[(df_test_sin_repes["G"] >= umbral_partidos) & (df_test_sin_repes["MP"] >= umbral_minutos)]
    else:
        df_test_umbral = df_test_sin_repes
    
    # --------------- SELECCION COLUMNAS --------------- #
    cols_a_usar = column_sets[modo_columnas]
    
    df_train_columnas = df_train_umbral[cols_a_usar].dropna()
    df_test_columnas  = df_test_umbral[cols_a_usar].dropna()

    if hay_fila_total_entrenamiento:
        df_train_columnas = df_train_columnas[:-1]
    if hay_fila_total_test:
        df_test_columnas  = df_test_columnas[:-1]
    
    # --------------- CONVERSION A NUMPY --------------- #
    X_train_raw = df_train_columnas.to_numpy()
    X_test_raw  = df_test_columnas.to_numpy()
    #X_total_raw = pd.read_csv(nombre_set_entrenamiento, encoding="utf-8")            # Para normalizar sin filtro de umbrales
    #X_total_raw = filtrar_fila_total(X_total_raw)[cols_a_usar].dropna().to_numpy()   # Para normalizar sin filtro de umbrales

    # --------------- NORMALIZACIÓN --------------- #
    if normalizar_datos:
        if modo_normalizacion == "zscore":
            mu    = X_train_raw.mean(axis=0)
            sigma = X_train_raw.std(axis=0) 
            sigma[sigma == 0] = 1e-8 # No queremos dividir entre 0 

            umbral_std = 1e-4 # No nos interesan estadísticas con desviación muy pequeña
            columnas_validas = sigma > umbral_std

            mu    = mu[columnas_validas]
            sigma = sigma[columnas_validas]

            X_train = (X_train_raw[:,columnas_validas] -  mu) / sigma
            X_test  = (X_test_raw[:,columnas_validas] -  mu) / sigma
        
        elif modo_normalizacion == "minmax":
            min_vals = X_train_raw.min(axis=0)
            max_vals = X_train_raw.max(axis=0)
            
            rango = max_vals - min_vals
            rango[rango == 0] == 1e-8 # No queremos dividir entre 0

            X_train = (X_train_raw - min_vals) / rango
            X_test  = (X_test_raw -min_vals) / rango
   
    else:
        X_train = X_train_raw
        X_test  = X_test_raw
    
    # --------------- PREPARACION TARGETS --------------- #
    if modo_autoencoder:
        Y_train = X_train 
        Y_test  = X_test
    else: # Preparar distintos tipos targets (por ahora solo pos)
        if modo_targets == "pos":
            posiciones_train_umbral = df_train_umbral["Pos"].tolist()
            posiciones_test_umbral  = df_test_umbral["Pos"].tolist()

            if hay_fila_total_entrenamiento:
                posiciones_train_umbral = posiciones_train_umbral[:-1]
            if hay_fila_total_test:
                posiciones_test_umbral = posiciones_test_umbral[:-1]
            
            Y_train = [mapping_pos_onehot.get(pos) for pos in posiciones_train_umbral]
            Y_test  = [mapping_pos_onehot.get(pos) for pos in posiciones_test_umbral]
        
        else:
            raise ValueError("No se reconoce el modo para los targets")
    # Nos aseguramos que los datos no contengan None
    cleaned_train = [(x, y) for x, y in zip(X_train, Y_train) if x is not None and y is not None]
    cleaned_test = [(x, y) for x, y in zip(X_test, Y_test) if x is not None and y is not None]

    X_train, Y_train = zip(*cleaned_train)
    X_test, Y_test = zip(*cleaned_test)

    # --------------- CONVERSION A TENSORES --------------- #
    X_train_tensor = torch.tensor(np.array(X_train), dtype=torch.float32)
    Y_train_tensor = torch.tensor(np.array(Y_train), dtype=torch.float32)
    X_test_tensor  = torch.tensor(np.array(X_test),  dtype=torch.float32)
    Y_test_tensor  = torch.tensor(np.array(Y_test),  dtype=torch.float32)
    if torch.isnan(X_train_tensor).any() or torch.isnan(Y_train_tensor).any():
        raise ValueError( "El entrenamiento contiene NaNs")
    if torch.isnan(X_test_tensor).any() or torch.isnan(Y_test_tensor).any():
        raise ValueError( "El test contiene NaNs")
    
    # --------------- ETIQUETADO --------------- #
    if modo_etiquetado is not None:
        if modo_etiquetado == "posicion":
            etiquetas_train = df_train_umbral["Pos"].tolist()
            etiquetas_test  = df_test_umbral["Pos"].tolist()
            if hay_fila_total_entrenamiento:
                etiquetas_train = etiquetas_train[:-1]
            if hay_fila_total_test:
                etiquetas_test = etiquetas_test[:-1]

        else:
            print(f"\nAVISO: No se reconoce el modo de etiquetado {modo_etiquetado}.\n")
            etiquetas_train = None
            etiquetas_test  = None
    else:
        etiquetas_train = None
        etiquetas_test  = None
    

    return X_train_tensor, Y_train_tensor, etiquetas_train, X_test_tensor, Y_test_tensor, etiquetas_test

