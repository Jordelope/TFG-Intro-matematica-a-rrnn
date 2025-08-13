import torch
import torch.nn.functional as F
import numpy
from MLP_mejoras import MLP, guardar_MLP, cargar_MLP
from Procesamiento_datos_modular import Xs_entrenamiento_def, Xs_test_def, Ys_entrenamiento_def, Ys_test_def
from Autoencoder_crear_mejoras import Autoencoder, guardar_autoencoder, cargar_autoencoder

## ARQUITECTURA de las redes ##
input_sz = 18        # Dimension de entradas
dim_lat = 2          # Dimension de salidas
estructura_oct_enc = [4,5,6]     # Capas ocultas
estructura_oct_dec = estructura_oct_enc[::-1]
lista_activaciones = None         # Lista de activación oculta (None = [None,...,None])
nombre_archivo_red = r"redes_disponibles\mejoras\test_guardado_autoencod.json"  # Archivo donde se guarda la red



## CARGAMOS los datos de entrenamiento y test ##
xs = Xs_entrenamiento_def
dato = xs[133]


if __name__ == "__main__":
    
    # Instanciación de la red
    encoder = MLP(input_sz, dim_lat, estructura_oct_enc, lista_activaciones)
    decoder = MLP(dim_lat, input_sz, estructura_oct_dec, lista_activaciones)
    autoencoder1 = Autoencoder(encoder,decoder)

    print(f"Hemos creado el modelo original {nombre_archivo_red}.")

    primera_pasada = autoencoder1(dato)
    print(f"La red original sobre el dato da : {primera_pasada}\n")


    guardar_autoencoder(autoencoder1,nombre_archivo_red)


    autoencoder2 = cargar_autoencoder(nombre_archivo_red)
    segunda_pasada = autoencoder2(dato)
    print(f"La red cargada sobre el dato da: {segunda_pasada}\n")
    
    if primera_pasada.detach().numpy().all() == segunda_pasada.detach().numpy().all():
        print("Parece que todo va BIEN.")
    else:
        print("ALGO FALLA")
