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



import pandas as pd

def completar_datasets(dataset1, dataset2, salida, clave="Player-additional",primera_fila=0):
    """
    Combina dos datasets de la misma temporada (mismos jugadores, diferentes estadísticas).
    - Conserva las columnas del primer dataset en caso de duplicados.
    - Une por la columna clave (por defecto 'Player-additional').
    - Emite avisos si se eliminaron columnas duplicadas.
    
    Parámetros:
    - dataset1 (str): ruta del primer CSV (columna principal).
    - dataset2 (str): ruta del segundo CSV (estadísticas adicionales).
    - salida (str): ruta donde guardar el dataset combinado.
    - clave (str): columna clave para emparejar jugadores (default: 'Player-additional').
    """
    
    # Cargar datasets
    df1 = pd.read_csv(dataset1)
    df2 = pd.read_csv(dataset2,header=primera_fila)  # salta la primera fila de categorías

    # Normalizar nombres de clave (por si acaso)
    if clave not in df1.columns:
        raise ValueError(f"'{clave}' no está en {dataset1}")
    if clave not in df2.columns:
        raise ValueError(f"'{clave}' no está en {dataset2}")

    # Detectar columnas duplicadas
    columnas_comunes = [c for c in df1.columns if c in df2.columns and c != clave]
    if columnas_comunes:
        print(f"⚠ Aviso: Se encontraron columnas duplicadas {columnas_comunes}. "
              f"Se conservarán las del primer dataset ({dataset1}).")
        # Eliminar duplicadas del segundo dataset
        df2 = df2.drop(columns=columnas_comunes)

    # Hacer merge en base a la clave
    combinado = pd.merge(df1, df2, on=clave, how="inner")

    # Guardar en CSV
    combinado.to_csv(salida, index=False)
    print(f"Se ha completado {dataset1} con {dataset2} en el nuevo {salida}.")
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
lista_datasets_shtg = [ r"datasets\nba\nba19_20_shooting.csv",   # Datasets pergame
                        r"datasets\nba\nba20_21_shooting.csv",
                        r"datasets\nba\nba21_22_shooting.csv",
                        r"datasets\nba\nba22_23_shooting.csv",
                        r"datasets\nba\nba23_24_shooting.csv",
                        r"datasets\nba\nba24_25_shooting.csv"
                        ]  
lista_datsets_adv = [r"datasets\nba\nba19_20_advanced.csv",   # Datasets pergame
                     r"datasets\nba\nba20_21_advanced.csv",
                     r"datasets\nba\nba21_22_advanced.csv",
                     r"datasets\nba\nba22_23_advanced.csv",
                     r"datasets\nba\nba23_24_advanced.csv",
                     r"datasets\nba\nba24_25_advanced.csv"
                    ] 

archivos_salida_temporadas = [r"datasets\nba\nba19_20_completo.csv",   # Datasets pergame
                              r"datasets\nba\nba20_21_completo.csv",
                              r"datasets\nba\nba21_22_completo.csv",
                              r"datasets\nba\nba22_23_completo.csv",
                              r"datasets\nba\nba23_24_completo.csv",
                              r"datasets\nba\nba24_25_completo.csv"
                    ]

archivo_salida_pg = r"datasets\nba\combined19_25_pergame_filtered.csv"  

combinar_pergame = False
completar_temporadas = True

if __name__=="__main__":

    if combinar_pergame:
        combinar_varios_datasets_filtrando(lista_datsets_pg,archivo_salida_pg)
    
    if completar_temporadas :
        for i in range(len(archivos_salida_temporadas)):
            completar_datasets(lista_datsets_pg[i] , lista_datsets_adv[i] , archivos_salida_temporadas[i])
            completar_datasets(archivos_salida_temporadas[i] , lista_datasets_shtg[i] , archivos_salida_temporadas[i],"Player-additional",1)

            