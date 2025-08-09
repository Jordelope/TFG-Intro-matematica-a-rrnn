"""

Primero entrenamos para el problema de predecir la posicion segun estadisticas.
Para un problema diferente, cambiar los vectores objetivo de entrenamiento y test.

"""
import pandas as pd
import numpy as np
from collections import Counter
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


## Parametros a ajustar ##

hacer_prints = True #<-- esto sigue siendo util?
modo_autoencoder = True # El autoencoder usara entrada=salida

hay_fila_totales_entrenamiento = False
hay_fila_totales_test = True

quitar_ruido_test = False

modos_posibles =["completo", "solo_volumen", "eficiencia_pura", "reducido"]
modo_columnas = "solo_volumen"

umbral_partidos = 10
umbral_minutos = 10

normalizar_datos = True
modos_norm_posibles = ["zscore", "minmax"]
modo_normalizacion = "zscore"

#--------------------------------------------------------------------------------------------------


## Filtramos jugadores repetidos(2 equipos misma temp) ##
def filtrar_fila_total(df):
    """
    Si un jugador tiene varias filas (por haber jugado en varios equipos),
    esta función conserva solo la fila con 'Team' == '2TM' (total de la temporada),
    y elimina las demás. Si no hay '2TM', se queda con la primera.
    """
    resultado = []
    for nombre, grupo in df.groupby("Player"):
        if len(grupo) == 1:
            resultado.append(grupo)
        else:
            total_row = grupo[grupo["Team"] == "2TM"]
            if not total_row.empty:
                resultado.append(total_row)
            else:
                resultado.append(grupo.iloc[:1])  # si no hay '2TM', coge la primera fila
    return pd.concat(resultado)

datos_original_entrenamiento = pd.read_csv(nombre_set_entrenamiento, encoding="utf-8")
datos_entrenamiento_sin_rep = filtrar_fila_total(datos_original_entrenamiento)

datos_original_test = pd.read_csv(nombre_set_test, encoding="utf-8")


## Filtramos jugadores que tengan pocos G/MP jugados (añaden ruido al entrenamiento) ##
datos_entrenamiento_sin_ruido = datos_entrenamiento_sin_rep[
    (datos_entrenamiento_sin_rep["G"] >= umbral_partidos) &
    (datos_entrenamiento_sin_rep["MP"] >= umbral_minutos)
]

if quitar_ruido_test:
    datos_test = datos_original_test[
        (datos_original_test["G"] >= umbral_partidos) &
        (datos_original_test["MP"] >= umbral_minutos)
    ]
else:
    datos_test = datos_original_test


## Filtramos estadisticas que nos interesan ##
cols_posibles = "Rk,Player,Age,Team,Pos,G,GS,MP,FG,FGA,FG%,3P,3PA,3P%,2P,2PA,2P%,eFG%,FT,FTA,FT%,ORB,DRB,TRB,AST,STL,BLK,TOV,PF,PTS,Awards,Player-additional"
column_sets = { # Configuraciones posibles de columnas a usar
    "completo": [
        "G", "GS", "MP",
        "FG", "FG%", "3P", "3P%", "2P", "2P%", "eFG%",
        "FT", "FT%",
        "ORB", "DRB", "AST", "STL", "BLK", "TOV", "PF"
    ],
    "solo_volumen": [
        "G", "GS", "MP",
        "FG","FGA", "3P", "3PA", "2P","2PA",
        "FT", "FTA",
        "ORB", "DRB", "AST", "STL", "BLK", "TOV", "PF"
    ],
    "eficiencia_pura": [
        "FG%", "3P%", "2P%", "eFG%", "FT%"
    ],
    "reducido": [
        "MP", "FG", "3P", "FT", "AST", "REB", "TOV"
    ]
}

cols_a_usar = column_sets[modo_columnas]

datos_totales_filtrados = datos_entrenamiento_sin_rep[cols_a_usar]
datos_entrenamiento_filtrados = datos_entrenamiento_sin_ruido[cols_a_usar]
datos_test_filtrados = datos_test[cols_a_usar]

datos_totales_filtrados = datos_totales_filtrados .dropna(subset=cols_a_usar)            # Eliminamos filas con datos que faltan ( si las hay)
datos_entrenamiento_filtrados = datos_entrenamiento_filtrados.dropna(subset=cols_a_usar) # Eliminamos filas con datos que faltan ( si las hay)
datos_test_filtrados = datos_test_filtrados.dropna(subset=cols_a_usar)                   # Eliminamos filas con datos que faltan ( si las hay)


##  Establecemos los vectores Xs_input,  Xs_filtrados y Xs_input test ##

Xs_filtrados = datos_totales_filtrados.to_numpy()                          # Para normalizar tomamos todos los datos
if hay_fila_totales_entrenamiento:                                         # Eliminamos la ultima filasi es la de totales
    Xs_filtrados = datos_entrenamiento_filtrados[:-1].to_numpy()                 
                                                                                  
Xs_input_entrenamiento = datos_entrenamiento_filtrados.to_numpy()          # Para entrenar excluimos ruido
if hay_fila_totales_entrenamiento:                                         # Eliminamos la ultima fila si es la de totales
    Xs_input_entrenamiento = datos_entrenamiento_sin_ruido[:-1].to_numpy()

Xs_input_test = datos_test_filtrados.to_numpy()
if hay_fila_totales_test:
    Xs_input_test = datos_test_filtrados[:-1].to_numpy()


## Establecemos los vectores objetivo del entrenamiento y test ##
mapping_pos = { ## Diccionario Posicion -> Vector ##
    "PG": [1,0,0,0,0],
    "SG": [0,1,0,0,0],
    "SF": [0,0,1,0,0],
    "PF": [0,0,0,1,0],
    "C" : [0,0,0,0,1] 
    }

posiciones_entrenamiento = datos_entrenamiento_sin_ruido["Pos"].tolist()           # Tomamos lista posiciones
if hay_fila_totales_entrenamiento:
    posiciones_entrenamiento = datos_entrenamiento_sin_ruido["Pos"].tolist()[:-1]  # Tomamos lista posiciones

posiciones_test = datos_test["Pos"].tolist()           # Tomamos lista posiciones
if hay_fila_totales_test:
    posiciones_test = datos_test["Pos"].tolist()[:-1]  # Tomamos lista posiciones

target_entrenamiento = list(map(lambda pos : mapping_pos.get(pos), posiciones_entrenamiento)) # Convertimos a vectores one-hot
target_test = list(map(lambda pos: mapping_pos.get(pos), posiciones_test))                    # Convertimos a vectores one-hot


#----------------------------------------------------------------------------------------------------------------------------------------------


### NORMALIZACION DATOS ###

if normalizar_datos:
    if modo_normalizacion == "zscore":
        ## Normalización estándar (media 0, varianza 1) ##
        # Tomamos (mu,sgma) de los datos TOTALES, aunque luego entrenemos con menos jugadores
        mu = Xs_filtrados.mean(axis=0)
        sigma = Xs_filtrados.std(axis=0)

        if hacer_prints: 
            for i, (m, s) in enumerate(zip(mu, sigma)):
                print(f"Columna {i}: media = {m:.3f}, std = {s:.6f}")
        
        sigma[sigma == 0] = 1e-8 # Evitar división por cero en columnas constantes

        # Filtramos columnas con std casi 0 
        umbral_std = 1e-4  # AJUSTABLE según el rango de datos
        columnas_validas = sigma > umbral_std

        # Aplicamos el filtrado a X, test y a estadísticas
        Xs_input = Xs_input_entrenamiento[:, columnas_validas]   
        test_input = Xs_input_test[:, columnas_validas]
        mu = mu[columnas_validas]
        sigma = sigma[columnas_validas]

        if hacer_prints: 
            print(f"\n{sum(~columnas_validas)} columnas eliminadas por tener desviación muy baja.")
            print(f"{sum(columnas_validas)} columnas conservadas.")


        # Normalizamos entrenamiento y test
        Xs_input_norm = (Xs_input - mu) / sigma
        test_input_norm = (test_input - mu) / sigma # Normalizamos test usando estadísticas del entrenamiento

    elif modo_normalizacion == "minmax":
        # Tomamos (mu,sgma) de los datos TOTALES, aunque luego entrenemos con menos jugadores
        min_vals = Xs_filtrados.min(axis=0)
        max_vals = Xs_filtrados.max(axis=0)
        rango = max_vals - min_vals
        rango[rango == 0] = 1e-8

        Xs_input_norm = (Xs_input_entrenamiento - min_vals) / rango
        test_input_norm = (Xs_input_test - min_vals) / rango
    
    #Limpieza
    Xs_input_norm = [x for x, y in zip(Xs_input_norm, target_entrenamiento) if y is not None]
    target_entrenamiento = [y for y in target_entrenamiento if y is not None]
    if np.isnan(Xs_input_norm).any():
        print("[ERROR] Datos normalizados contienen NaNs.")

    ## Valores definitivos de entrenamiento para importar (en torch)##
    Xs_entrenamiento_def = torch.tensor(np.array(Xs_input_norm), dtype=torch.float32)

    Ys_entrenamiento_def = torch.tensor(np.array(target_entrenamiento), dtype=torch.float32)

    Xs_test_def = torch.tensor(np.array(test_input_norm), dtype=torch.float32)
    Ys_test_def = torch.tensor(np.array(target_test), dtype=torch.float32)

    if modo_autoencoder:
        Ys_entrenamiento_def = Xs_entrenamiento_def
        Ys_test_def = Xs_test_def
    
else :
    
    #Limpieza
    Xs_filtrados = [x for x, y in zip(Xs_filtrados, target_entrenamiento) if y is not None]
    target_entrenamiento = [y for y in target_entrenamiento if y is not None]
    if np.isnan(Xs_filtrados).any():
        print("[ERROR] Datos normalizados contienen NaNs.")

    Xs_entrenamiento_def = torch.tensor(np.array( Xs_filtrados) , dtype=torch.float32)
    Ys_entrenamiento_def = torch.tensor(np.array(target_entrenamiento), dtype=torch.float32)

    Xs_test_def = torch.tensor( np.array(Xs_input_test) , dtype=torch.float32)
    Ys_test_def = torch.tensor(np.array(target_test), dtype=torch.float32)

    if modo_autoencoder:
        Ys_entrenamiento_def = Xs_entrenamiento_def
        Ys_test_def = Xs_test_def
    



### REVISION ERRORES ###
if torch.isnan(Xs_entrenamiento_def).any():
    print("[ERROR] Xs_entrenamiento_def contiene NaNs")

for i, y in enumerate(target_test):
    if y is None:
        print(f"[ERROR] Posición inválida: {posiciones_test[i]}")
