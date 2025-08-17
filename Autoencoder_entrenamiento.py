import torch
import torch.nn.functional as F
from MLP import MLP, cargar_MLP, guardar_MLP
from Autoencoder import Autoencoder, cargar_autoencoder, guardar_autoencoder
from Procesar_datos import procesar_datos

"""
Duda existencial: loss_f(las de torch) calculan bien la perdida si le estamos pasando el batch no?
"""

## Funciones relevantes ##



## DATOS de red a entrenar ##

archivo_encod = r"redes_disponibles\visual_pruebas_dim6_enc.json"
archivo_decod = r"redes_disponibles\visual_pruebas_dim6_dec.json"
archivo_autoencoder = r"redes_disponibles\visual_pruebas_dim6_autoenc.json"

## OPCIONES de guardado ##

save_after_training = True  # En caso de True: se guarda cuando mejora el error respecto 
override_guardado = True   # En caso de True: se guarda aunque no mejore el error (si el anterior es True)

sobreescribir_submodelos = True # En caso de True: Se sobreescriben archivos de encoder y decoder.

## HIPERPARAMETROS de entrenamiento ##

stp_n = 1000000     # Número de pasos de entrenamiento
stp_sz = 0.0005    # Tamaño del paso (learning rate)
batch_sz = None  # Tamaño del batch (por defecto si es None, todo el dataset)

loss_f = F.mse_loss # Función de pérdida
beta = 0.005
lambda_l2 = 0.001


## DATOS de entrenamiento y test ##
xs_train,ys_train,etiquetas_train,_,xs_test,ys_test,etiquetas_test = procesar_datos(archivo_set_train="datasets/nba_pergame_24_full.csv",
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
