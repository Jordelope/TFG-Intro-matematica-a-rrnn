import torch
import torch.nn.functional as F
import numpy
from MLP_mejoras import MLP, guardar_MLP, cargar_MLP
from Procesamiento_datos_modular import Xs_entrenamiento_def, Xs_test_def, Ys_entrenamiento_def, Ys_test_def


## ARQUITECTURA de la red ##
input_sz = 18        # Dimension de entradas
out_sz = 2          # Dimension de salidas
estructura_oct = [4,5,6]     # Capas ocultas
lista_activaciones = None         # Lista de activación oculta (None = [None,...,None])
nombre_archivo_red = r"redes_disponibles\mejoras\test_guardado_mlp.json"  # Archivo donde se guarda la red



## CARGAMOS los datos de entrenamiento y test ##
xs = Xs_entrenamiento_def
dato = xs[133]


if __name__ == "__main__":
    
    # Instanciación de la red
    NN1 = MLP(input_sz, out_sz, estructura_oct, lista_activaciones)
    print(f"Hemos creado el modelo original {nombre_archivo_red}.")

    primera_pasada = NN1(dato)
    print(f"La red original sobre el dato da : {primera_pasada}\n")


    guardar_MLP(NN1,nombre_archivo_red)


    NN2 = cargar_MLP(nombre_archivo_red)
    segunda_pasada = NN2(dato)
    print(f"La red cargada sobre el dato da: {segunda_pasada}\n")
    
    if primera_pasada.numpy().all() == segunda_pasada.numpy().all():
        print("Parece que todo va BIEN.")
    else:
        print("ALGO FALLA")
