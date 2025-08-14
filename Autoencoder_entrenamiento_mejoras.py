import torch
import torch.nn.functional as F
from MLP_mejoras import MLP, cargar_MLP, guardar_MLP
from Autoencoder_mejoras import Autoencoder, cargar_autoencoder, guardar_autoencoder
from Procesamiento_datos_modular import Xs_entrenamiento_def, Ys_entrenamiento_def, Xs_test_def, Ys_test_def

"""
Duda existencial: loss_f(las de torch) calculan bien la perdida si le estamos pasando el batch no?
"""

## Funciones relevantes ##



## DATOS de red a entrenar ##

archivo_encod = r"redes_disponibles\mejoras\encoder_pruebas.json"
archivo_decod = r"redes_disponibles\mejoras\decoder_pruebas.json"
archivo_autoencoder = r"redes_disponibles\mejoras\autoencoder_pruebas.json"

## OPCIONES de guardado ##

save_after_training = True  # En caso de True: se guarda cuando mejora el error respecto 
override_guardado = False   # En caso de True: se guarda aunque no mejore el error (si el anterior es True)

sobreescribir_submodelos = True # En caso de True: Se sobreescriben archivos de encoder y decoder.

## HIPERPARAMETROS de entrenamiento ##

stp_n = 10     # Número de pasos de entrenamiento
stp_sz = 0.005    # Tamaño del paso (learning rate)
batch_sz = None  # Tamaño del batch (por defecto si es None, todo el dataset)

loss_f = F.mse_loss # Función de pérdida


## DATOS de entrenamiento y test ##
xs_train = Xs_entrenamiento_def
ys_train = Ys_entrenamiento_def

xs_test = Xs_test_def
ys_test = Ys_test_def



#------------------------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    
    ## CARGAR red ##
    print(f"\nSe va entrenar el modelo '{archivo_autoencoder}'.")
    NN = cargar_autoencoder(archivo_autoencoder)

    ## AVISOS ##
    if  save_after_training:
        if override_guardado:
            print(f"\nAVISO: El modelo '{archivo_autoencoder}' se va a guardar aunque empeore el error.")
        else:
            print(f"\nAVISO: El modelo '{archivo_autoencoder}' se va a guardar.")
        if sobreescribir_submodelos:
            print(f"\nAVISO: Se van a sobrescribir el encoder y decoder.\n")
    else:
        print(f"\nAVISO: El modelo '{archivo_autoencoder}' no se va a guardar.")


    ## ERROR INICIAL sobre el test ##
    with torch.no_grad():
        pred_test_init = NN(xs_test)
        loss_init = loss_f(pred_test_init, ys_test) 
    print(f"\nEl modelo '{archivo_autoencoder}' tiene una perdida inicial sobre el test: {loss_init}.")


    ## ENTRENAMIENTO ##
    print(f"\nIniciamos entrenamiento de {stp_n} pasos de el modelo '{archivo_autoencoder}'.\n") 
    NN.train_model(xs_train,stp_n,stp_sz,loss_f,batch_sz)


    ## ERROR FINAL sobre el test ##
    with torch.no_grad():
        pred_test_fin = NN(xs_test)
        loss_final = loss_f(pred_test_fin, ys_test) 
    print(f"\nEl modelo '{archivo_autoencoder}' tiene una perdida final sobre el test: {loss_final}.\n")

    
    ## GUARDADO de la red en su archivo original ##
    if save_after_training:
        if loss_final < loss_init or override_guardado:
            
            print( f"El error del modelo '{archivo_autoencoder}' sobre el test ha mejorarado y por tanto la actualizamos.\n")
            guardar_autoencoder(NN,archivo_autoencoder)
            
            if sobreescribir_submodelos:
                guardar_MLP(NN.encoder,archivo_encod)
                guardar_MLP(NN.decoder,archivo_decod)
        else:
            print( f"El error del modelo '{archivo_autoencoder}' sobre el test no ha mejorarado y por tanto NO la actualizamos.\n")
    else:
        print(f"Se ha decidido NO guardar la actualizacion del modelo '{archivo_autoencoder}.'\n")
