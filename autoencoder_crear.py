import torch
import torch.nn.functional as F
import random
from MLP import MLP, cargar_MLP, guardar_MLP
from proc_datos_modular import Xs_entrenamiento_def, Xs_test_def, Ys_entrenamiento_def, Ys_test_def
from Autoencoder import Autoencoder, guardar_autoencoder, cargar_autoencoder


"""
Pasos proximo dia:
- Añadir un test de rendimiento
- Decidir cuales son las funciones de activacion mas adeccuadas
- Definir una buena estructura de autoencoder y entrenar
- Crear un primer autoencoder y ver como funciona el encoder
- crear un clasiicador a partir del encoder y ver si hay que entrenarlo junto al encoder o solo el clasificador
"""

## NOMBRE archivos de MLP (si ya los tenemos) y OPCIONES de entrenado y guardado ##
existen_MLP = False
archivo_encod = r"redes_disponibles\encoder_prueba.json"
archivo_decod = r"redes_disponibles\decoder_prueba.json"
archivo_autoencoder = r"redes_disponibles\autoencoder_prueba.json"

train_autoencoder = False

save_autoencoder = True
save_decoder = True
save_encoder = True


## ESTRUCTURA autoencoder (si no tenemos los MLP) ##

input_sz = len(Xs_entrenamiento_def[0])            # Número de entradas
lat_spc_dim = 8                  # Dimension espacio latente(salida encoder, entrada decoder)

estructura_encod = [36, 18 , 10]      # Capas ocultas encoder
estructura_decod = [10, 18, 36 ]      # Capas ocultas decoder

f_out_encod = None                # Función de activación de salida  encoder    (None = por defecto lineal en MLP)
f_oculta_encod =  F.relu           # Función de activación capas ocultas encoder (None = por defecto linealen MLP)
f_out_decod = None                  # Función de activación de salida  decoder    (None = por defecto lineal en MLP)
f_oculta_decod = F.relu               # Función de activación capas ocultas decoder (None = por defecto lineal en MLP)


## HIPERPARAMETROS de entrenamiento ##

stp_n = 300     # Número de pasos de entrenamiento
stp_sz = stp_sz = 0.001    # Tamaño del paso (learning rate)
batch_sz = 8  # Tamaño del batch (por defecto, todo el dataset)

loss_f = F.mse_loss # Función de pérdida


## Elegimos los datos de entrenamiento ##
xs = Xs_entrenamiento_def


if __name__ == "__main__":
    
    if not save_autoencoder and not save_encoder and not save_decoder:
        print("AVISO: No se guardará ninguna red.")
    if train_autoencoder:
        print("AVISO: El autoencoder se entrenará.")
    else:
        print("AVISO: El autoencoder no se entrenará")
    

    # Cargamos los MLP o los creamos si no existen
    if existen_MLP:
        # Cargar MLP existentes
        encoder = cargar_MLP(archivo_encod)
        decoder = cargar_MLP(archivo_decod)
    else:
        # Crear nuevos MLP para encoder y decoder
        encoder = MLP(input_sz, lat_spc_dim, estructura_encod, f_out_encod, f_oculta_encod)
        decoder = MLP(lat_spc_dim, input_sz, estructura_decod, f_out_decod, f_oculta_decod)
    
    
    # Crear autoencoder
    autoencoder = Autoencoder(encoder, decoder)
    

    if train_autoencoder:
        if torch.isnan(Xs_entrenamiento_def).any():
            print("[ERROR] Xs_entrenamiento_def contiene NaNs")
        if torch.isinf(Xs_entrenamiento_def).any():
            print("[ERROR] Xs_entrenamiento_def contiene infinitos")

        print(f"Iniciamos entrenamiento de {stp_n} pasos del autoencoder '{archivo_autoencoder}'.\n")
        autoencoder.train_model(xs, stp_n, stp_sz, loss_f, batch_sz)
    

    # Guardamos las redes segun eleccion
    if save_autoencoder:
        guardar_autoencoder(autoencoder, archivo_autoencoder)
    if save_encoder :
        guardar_MLP(encoder, archivo_encod)
    if save_decoder:
        guardar_MLP(decoder, archivo_decod) 
