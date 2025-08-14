import torch
import torch.nn.functional as F
import random
from MLP_mejoras import MLP, cargar_MLP, guardar_MLP
from Procesamiento_datos_modular import Xs_entrenamiento_def, Xs_test_def, Ys_entrenamiento_def, Ys_test_def
from Autoencoder_mejoras import Autoencoder, guardar_autoencoder, cargar_autoencoder


"""
PENDIENTE:

- Decidir cuales son las funciones de activacion mas adeccuadas
- Definir una buena estructura de autoencoder y entrenar

"""

## NOMBRE archivos de MLP (si ya los tenemos) y OPCIONES de entrenado y guardado ##
existen_MLP = True
archivo_encod = r"redes_disponibles\mejoras\encoder_pruebas.json"
archivo_decod = r"redes_disponibles\mejoras\decoder_pruebas.json"
archivo_autoencoder = r"redes_disponibles\mejoras\autoencoder_pruebas.json"

train_autoencoder = True

save_autoencoder = True
save_decoder = True
save_encoder = True


## ESTRUCTURA autoencoder (si no tenemos los MLP) ##

input_sz = len(Xs_entrenamiento_def[0])            # Número de entradas
lat_spc_dim = 3                                    # Dimension espacio latente(salida encoder, entrada decoder)

estructura_encod = [36, 18 , 10]               # Capas ocultas encoder
estructura_decod = estructura_encod[::-1]      # Capas ocultas decoder

lista_act_encod = [F.relu for i in range(len(estructura_encod))] + [None] # Funciones activacion encoder (None = [None,...,None] por defecto lineal en MLP)
lista_act_decod = [F.relu for i in range(len(estructura_decod))] + [None] # Funciones activacion encoder (None = [None,...,None] por defecto lineal en MLP)

## HIPERPARAMETROS de entrenamiento ##

stp_n = 10     # Número de pasos de entrenamiento
stp_sz = stp_sz = 0.001    # Tamaño del paso (learning rate)
batch_sz = None  # Tamaño del batch (por defecto, todo el dataset)

loss_f = F.mse_loss # Función de pérdida


## DATOS de entrenamiento y test ##
xs_train = Xs_entrenamiento_def
ys_train = Ys_entrenamiento_def
xs_test = Xs_test_def
ys_test = Ys_test_def

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
        encoder = cargar_MLP(archivo_encod)
        decoder = cargar_MLP(archivo_decod)
    else:
        # Crear nuevos MLP para encoder y decoder
        encoder = MLP(input_sz, lat_spc_dim, estructura_encod, lista_act_encod)
        decoder = MLP(lat_spc_dim, input_sz, estructura_decod, lista_act_decod)
    
    
    # Crear autoencoder
    autoencoder = Autoencoder(encoder, decoder)
    

    if train_autoencoder:
        if torch.isnan(Xs_entrenamiento_def).any():
            print("[ERROR] Xs_entrenamiento_def contiene NaNs")
        if torch.isinf(Xs_entrenamiento_def).any():
            print("[ERROR] Xs_entrenamiento_def contiene infinitos")

        ## ERROR INICIAL sobre el test ##
        with torch.no_grad():
            pred_test_init = autoencoder(xs_test)
            init_loss = loss_f(pred_test_init,ys_test) 
        print(f"\nEl modelo '{archivo_autoencoder}' tiene una perdida inicial sobre el test: {init_loss}.\n")


        print(f"\nIniciamos entrenamiento de {stp_n} pasos del autoencoder '{archivo_autoencoder}'.\n")
        autoencoder.train_model(xs_train, stp_n, stp_sz, loss_f, batch_sz)

        ## ERROR FINAL sobre el test ##
        with torch.no_grad():
            pred_test_fin = autoencoder(xs_test)
            loss_final = loss_f(pred_test_fin,ys_test) 
        print(f"\nEl modelo '{archivo_autoencoder}' tiene una perdida final sobre el test: {loss_final}.\n")

    

    # Guardamos las redes segun eleccion
    if save_autoencoder:
        guardar_autoencoder(autoencoder, archivo_autoencoder)
        if save_encoder :
            guardar_MLP(encoder, archivo_encod)
        if save_decoder:
            guardar_MLP(decoder, archivo_decod) 
