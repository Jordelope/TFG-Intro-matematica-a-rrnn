import torch
import torch.nn.functional as F
import random
from MLP import MLP, guardar_MLP, cargar_MLP
from Procesar_datos import procesar_datos
"""
crear_red_torch.py
-------------------

Este script permite crear, entrenar y guardar una red neuronal multicapa (MLP) utilizando la implementación matemática definida en el módulo MLP.py.
El usuario puede definir la arquitectura de la red, los parámetros de entrenamiento y decidir si desea entrenar y/o guardar la red resultante en un archivo JSON.

Estructura general del script:
- Carga los datos de entrenamiento y test desde procesar_datos_entrenamiento.py.
- Permite definir la arquitectura y funciones de activación de la red MLP.
- Permite entrenar la red con los parámetros elegidos (número de pasos, tamaño de batch, función de pérdida, etc).
- Permite guardar la red (estructura y pesos) en un archivo JSON para su posterior uso.
- Incluye utilidades para decodificar la salida de la red y evaluar su precisión sobre el conjunto de test.

"""
## Funciones relevantes ##





## ARQUITECTURA de la red ## (Prestar MUCHA ATENCION A FUNCIONES ACTIVACION)
input_sz = 18      # Número de entradas (asegurar que coincide con los datos que se va a usar)
out_sz = 5        # Número de salidas (asegurar que coincide con los datos que se va a usar)
estructura_oct = [18, 18]  # Capas ocultas
lista_act = None           # Lista de funciones de activación (asegurar compatible con estructura oct)
                           #(None =[None,...,None] por defecto en MLP)

nombre_archivo_red = r"redes_disponibles\mlp_prueba_desc.json"  # Archivo donde se guarda la red

## OPCIONES GUARDADO ##
save_new_NN = True         # ¿Guardar la red tras crearla?
entrenar_nueva_red = True # ¿Entrenar la red tras crearla?


## HIPERPARAMETROS de entrenamiento ##
stp_n = 10  # Número de pasos de entrenamiento
stp_sz = 0.00025 # Tamaño del paso (learning rate)
batch_sz = None  # Tamaño del batch (None = por defecto, todo el dataset)

loss_f = F.mse_loss # Función de pérdida


## Cargamos los datos de entrenamiento ##
xs_train, ys_train, etiquetas_train, xs_test, ys_test, etiquetas_test = procesar_datos(archivo_set_train="datasets/nba_pergame_24_full.csv",
                                                                                    archivo_set_test="datasets/nba_pergame_24_full.csv",
                                                                                    modo_autoencoder=False,
                                                                                    modo_columnas="solo_volumen",
                                                                                    modo_targets="pos",
                                                                                    modo_etiquetado="posicion",
                                                                                    normalizar_datos=True,
                                                                                    modo_normalizacion="zscore",
                                                                                    umbral_partidos=5,
                                                                                    umbral_minutos=5,
                                                                                    umbral_en_test=True,
                                                                                    hay_fila_total_entrenamiento=False,
                                                                                    hay_fila_total_test=True)



#-------------------------------------------------------------------------------------------------------------------------------



if __name__ == "__main__":
    # Instanciación de la red
    NN = MLP(input_sz, out_sz, estructura_oct, lista_act)
    print(f"\nHemos creado la red {nombre_archivo_red}.\n")

    if save_new_NN:
        print(f"\nAVISO: La red {nombre_archivo_red} se va a guardar.\n")
    else:
        print(f"\nAVISO: La red {nombre_archivo_red} no se va a guardar.\n")

    if entrenar_nueva_red:
        print(f"AVISO: La red {nombre_archivo_red} se va a entrenar.\n")
    else:
        print(f"AVISO: La red {nombre_archivo_red} no se va a entrenar.\n")

    # Entrenamiento de la red
    if entrenar_nueva_red:

        ## ERROR INICIAL sobre el test ##
        with torch.no_grad():
            pred_test_init = NN(xs_test)
            loss_init = loss_f(pred_test_init,ys_test)
        print(f"\nLa red '{nombre_archivo_red}' tiene una perdida inicial sobre el test: {loss_init.detach().numpy()}.\n")

        ## ENTRENAMIENTO ##
        print(f"\nIniciamos entrenamiento de {stp_n} pasos de la red '{nombre_archivo_red}'.\n")
        NN.train_model(xs_train, ys_train, stp_n, stp_sz, loss_f, batch_sz)

        ## ERROR FINAL sobre el test ##
        with torch.no_grad():
            pred_test_fin = NN(xs_test)
            loss_final = loss_f(pred_test_fin,ys_test)
        print(f"\nLa red '{nombre_archivo_red}' tiene una perdida final sobre el test: {loss_final}.\n")

    # Guardado de la red
    if save_new_NN:
        guardar_MLP(NN, nombre_archivo_red)
    else:
        print(f"Se ha decidido no guardar la red {nombre_archivo_red}.\n")
