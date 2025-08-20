import pandas as pd

def combinar2_datasets(dataset1, dataset2, salida):
    """
    Combina dos datasets CSV en uno solo sin eliminar duplicados.
    
    Parámetros:
    - dataset1 (str): ruta al primer CSV
    - dataset2 (str): ruta al segundo CSV
    - salida (str): ruta donde guardar el CSV combinado
    """
    # Leer ambos datasets
    df1 = pd.read_csv(dataset1)
    df2 = pd.read_csv(dataset2)

    # Concatenar sin eliminar duplicados
    combinado = pd.concat([df1, df2], ignore_index=True)

    # Guardar en un nuevo archivo
    combinado.to_csv(salida, index=False)

    return combinado



def combinar_varios_datasets(lista_datasets, salida):
    """
    Combina varios datasets CSV en uno solo sin eliminar duplicados.
    
    Parámetros:
    - lista_datasets (list): lista con las rutas de los CSV a combinar
    - salida (str): ruta donde guardar el CSV combinado
    """
    # Leer y acumular todos los datasets
    dataframes = [pd.read_csv(archivo) for archivo in lista_datasets]

    # Concatenar todos
    combinado = pd.concat(dataframes, ignore_index=True)

    # Guardar en un nuevo archivo
    combinado.to_csv(salida, index=False)

    return combinado




def combinar_varios_datasets_filtrando(lista_datasets, salida):
    """
    Combina varios datasets CSV en uno solo sin eliminar duplicados entre temporadas,
    pero dentro de cada dataset filtra a los jugadores con múltiples equipos,
    dejando solo la fila con '2TM', '3TM', '4TM', etc.
    Mantiene el orden original de 'Rk' dentro de cada dataset.
    
    Parámetros:
    - lista_datasets (list): lista con las rutas de los CSV a combinar
    - salida (str): ruta donde guardar el CSV combinado
    """
    dataframes = []

    for archivo in lista_datasets:
        df = pd.read_csv(archivo)


        # Filtrar duplicados de jugadores en varios equipos:
        filtrado = []
        for jugador, grupo in df.groupby("Player", sort=False):
            multi_team = grupo[grupo["Team"].str.contains("TM", na=False)]
            if not multi_team.empty:
                filtrado.append(multi_team.iloc[0])  # Guardamos la fila nTM
            else:
                filtrado.extend(grupo.to_dict("records"))  # Guardamos todas las demás filas

        df_filtrado = pd.DataFrame(filtrado)

        # Reordenar por Rk de nuevo (por si acaso al filtrar se alteró)
        if "Rk" in df_filtrado.columns:
            df_filtrado = df_filtrado.sort_values("Rk", ascending=True)

        dataframes.append(df_filtrado)

    # Concatenar todos los datasets ya filtrados (en el orden de la lista)
    combinado = pd.concat(dataframes, ignore_index=True)

    # Guardar en CSV
    combinado.to_csv(salida, index=False)

    return combinado


#---------------------------------------------------------------------------------------------------------------------------
lista_datsets_pg = [r"datasets\nba\nba19_20_pergame.csv",   # Datasets pergame
                    r"datasets\nba\nba20_21_pergame.csv",
                    r"datasets\nba\nba21_22_pergame.csv",
                    r"datasets\nba\nba22_23_pergame.csv",
                    r"datasets\nba\nba23_24_pergame.csv",
                    r"datasets\nba\nba24_25_pergame.csv"
                    ]
 
archivo_salida_pg = r"datasets\nba\combined19_25_pergame_filtered.csv"  

combinar_pergame = True

if __name__=="__main__":

    if combinar_pergame:
        combinar_varios_datasets_filtrando(lista_datsets_pg,archivo_salida_pg)