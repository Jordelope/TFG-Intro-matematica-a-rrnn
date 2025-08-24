import torch
import torch.nn.functional as F
import random
from MLP import MLP
from Autoencoder import Autoencoder
from Guardar_Cargar import guardar_modelo, cargar_modelo
from Procesar_datos import procesar_datos


## NOMBRE archivos (se tomaran los MLP si ya los tenemos) ##
existen_MLP = False
archivo_encod = r"redes_disponibles\encod_nba1.json" 
archivo_decod = r"redes_disponibles\decod_nba1.json" 
archivo_autoencoder = r"redes_disponibles\autoencoder_nba2_mejoras.json" 


## OPCIONES de entrenado y guardado ##
save_autoencoder = True
save_decoder = False
save_encoder = False

train_autoencoder = False

añadir_descripcion = True  # Opción de añadir una descripción
descripcion = " Autoencoder pequeno para probar no linealidad en las ocultas."

## ESTRUCTURA autoencoder (si no tenemos los MLP) ##

input_sz = 18            # Número de entradas
lat_spc_dim = 8                                    # Dimension espacio latente(salida encoder, entrada decoder)
#probar añadir una de 128 la primera(y ultima en decod)
estructura_encod = [64,32]               # Capas ocultas encoder
estructura_decod = estructura_encod[::-1]      # Capas ocultas decoder

lista_act_encod = [F.relu, torch.tanh] + [None] # Funciones activacion encoder (None = [None,...,None] por defecto lineal en MLP)
lista_act_decod = [torch.tanh, F.relu] + [None] # Funciones activacion encoder (None = [None,...,None] por defecto lineal en MLP)

## HIPERPARAMETROS de entrenamiento ##

stp_n = 50000     # Número de pasos de entrenamiento
stp_sz = 0.001    # Tamaño del paso (learning rate)
batch_sz = 32  # Tamaño del batch (por defecto, todo el dataset)

loss_f = F.mse_loss # Función de pérdida
beta = 0.0075
lambda_l2 = 0.001

## DATOS de entrenamiento y test ##
archivo_entrenamiento = r"datasets\nba\combined19_25_pergame_filtered.csv"
archivo_test = r"datasets\equipos\roster_hawks_pergame_25.csv" # Seria ideal poner de test datos que no hubiera visto
xs_train, ys_train, etiquetas_train, xs_test, ys_test, etiquetas_test = procesar_datos(archivo_set_train=archivo_entrenamiento,
                                                                                    archivo_set_test=archivo_test,
                                                                                    modo_autoencoder=True,
                                                                                    modo_columnas="solo_volumen",
                                                                                    modo_targets="pos",
                                                                                    modo_etiquetado="posicion",
                                                                                    normalizar_datos=True,
                                                                                    modo_normalizacion="zscore",
                                                                                    umbral_partidos=20,
                                                                                    umbral_minutos=15,
                                                                                    umbral_en_test=True,
                                                                                    hay_fila_total_entrenamiento=False,
                                                                                    hay_fila_total_test=True)


if __name__ == "__main__":
    
    if not save_autoencoder and not save_encoder and not save_decoder:
        print("AVISO: No se guardará ninguna red.\n")
    if train_autoencoder:
        print("AVISO: El autoencoder se entrenará.\n")
    else:
        print("AVISO: El autoencoder no se entrenará.\n")
    

    # Cargamos los MLP o los creamos si no existen
    if existen_MLP:
        # Cargar MLP existentes
        encoder = cargar_modelo(archivo_encod)
        decoder = cargar_modelo(archivo_decod)
    else:
        # Crear nuevos MLP para encoder y decoder
        encoder = MLP(input_sz, lat_spc_dim, estructura_encod, lista_act_encod)
        decoder = MLP(lat_spc_dim, input_sz, estructura_decod, lista_act_decod)
    
    
    # Crear autoencoder
    autoencoder = Autoencoder(encoder, decoder)
    

    if train_autoencoder:
        if torch.isnan(xs_train).any():
            print("[ERROR] Xs_entrenamiento_def contiene NaNs")
        if torch.isinf(xs_train).any():
            print("[ERROR] Xs_entrenamiento_def contiene infinitos")

        ## ERROR INICIAL sobre el test ##
        with torch.no_grad():
            pred_test_init = autoencoder(xs_test)
            init_loss = loss_f(pred_test_init,ys_test) 
        print(f"\nEl modelo '{archivo_autoencoder}' tiene una perdida inicial sobre el test: {init_loss}.\n")


        print(f"\nIniciamos entrenamiento de {stp_n} pasos del autoencoder '{archivo_autoencoder}'.\n")
        autoencoder.train_model(xs_train, stp_n, stp_sz, loss_f, batch_sz,beta,lambda_l2)

        ## ERROR FINAL sobre el test ##
        with torch.no_grad():
            pred_test_fin = autoencoder(xs_test)
            loss_final = loss_f(pred_test_fin,ys_test) 
        print(f"\nEl modelo '{archivo_autoencoder}' tiene una perdida final sobre el test: {loss_final}.\n")

    

    # Guardamos las redes segun eleccion
    if añadir_descripcion:
        autoencoder.description = descripcion
    if save_autoencoder:
        guardar_modelo(autoencoder, archivo_autoencoder)
        if save_encoder :
            guardar_modelo(encoder, archivo_encod)
        if save_decoder:
            guardar_modelo(decoder, archivo_decod) 
    else:
        print(f"Se ha decidido no guardar la red {archivo_autoencoder}.\n")

