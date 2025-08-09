import torch
from proc_datos_modular import procesar_datos

def test_procesar_datos_autoencoder():
    X_train, Y_train, X_test, Y_test = procesar_datos(
        nombre_set_entrenamiento="datasets/nba_pergame_24_full.csv",
        nombre_set_test="datasets/roster_hawks_pergame_25.csv",
        modo_columnas="solo_volumen",
        modo_normalizacion="zscore",
        normalizar_datos=True,
        modo_autoencoder=True,
        hacer_prints=False
    )

    assert not torch.isnan(X_train).any(), "X_train contiene NaNs"
    assert not torch.isnan(Y_train).any(), "Y_train contiene NaNs"
    assert X_train.shape == Y_train.shape, "En autoencoder, entrada y salida deben coincidir"
    print("[✓] Autoencoder: Datos procesados correctamente")

def test_procesar_datos_clasificacion():
    X_train, Y_train, X_test, Y_test = procesar_datos(
        nombre_set_entrenamiento="datasets/nba_pergame_24_full.csv",
        nombre_set_test="datasets/roster_hawks_pergame_25.csv",
        modo_columnas="solo_volumen",
        modo_normalizacion="zscore",
        normalizar_datos=True,
        modo_autoencoder=False,
        hacer_prints=False
    )

    assert not torch.isnan(X_train).any(), "X_train contiene NaNs"
    assert not torch.isnan(Y_train).any(), "Y_train contiene NaNs"
    assert X_train.shape[0] == Y_train.shape[0], "Número de muestras no coincide"
    assert Y_train.shape[1] == 5, "La salida de clasificación debe ser one-hot con 5 clases"
    print("[✓] Clasificación: Datos procesados correctamente")


if __name__ == "__main__":
    test_procesar_datos_autoencoder()
    test_procesar_datos_clasificacion()
