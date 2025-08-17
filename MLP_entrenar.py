import torch
import torch.nn.functional as F
import random
from MLP import MLP, guardar_MLP, cargar_MLP
from Procesar_datos import procesar_datos

## Funciones relevantes ##
def clasificacion(xs):
    return xs.argmax().item()  # Devuelve el índice de la clase con mayor probabilidad


## MODELO a entrenar ##

nombre_archivo_red = r"redes_disponibles\nuevo_mlp_pruebas.json"  # Archivo donde se guarda la red


## HIPERPARAMETROS de entrenamiento ##
stp_n = 5     # Número de pasos de entrenamiento
stp_sz = stp_sz = 0.001    # Tamaño del paso (learning rate)
batch_sz = None  # Tamaño del batch (por defecto, todo el dataset)

loss_f = F.mse_loss # Funcion de perdida


## OPCIONES de guardado ##  
save_after_training = True  # En caso de True: se guarda cuando mejora el error respecto 
override_guardado = True   # En caso de True: se guarda aunque no mejore el error (si el anterior es True)


## DATOS de test y entrenamiento ##
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




#-------------------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    
    ## CARGAR red ##
    print(f"\nSe va entrenar el modelo '{nombre_archivo_red}'.")
    NN = cargar_MLP(nombre_archivo_red)


    ## AVISOS ##
    if  save_after_training:
        if override_guardado:
            print(f"\nAVISO: El modelo '{nombre_archivo_red}' se va a guardar aunque empeore el error.")
        else:
            print(f"\nAVISO: El modelo '{nombre_archivo_red}' se va a guardar.")
    else:
        print(f"\nAVISO: El modelo '{nombre_archivo_red}' NO se va a guardar.")


    ## ERROR INICIAL sobre el test ##
    with torch.no_grad():
        pred_test_init = NN(xs_test)
        init_loss = loss_f(pred_test_init,ys_test) 
    print(f"\nEl modelo '{nombre_archivo_red}' tiene una perdida inicial sobre el test: {init_loss}")

    ## ENTRENAMIENTO ##
    print(f"\nIniciamos entrenamiento de {stp_n} pasos del  modelo '{nombre_archivo_red}'.\n") 
    NN.train_model(xs_train,ys_train,stp_n,stp_sz,loss_f,batch_sz)
    
    
    ## ERROR FINAL sobre el test ##
    with torch.no_grad():
        pred_test_fin = NN(xs_test)
        loss_final = loss_f(pred_test_fin,ys_test) 
    print(f"\nEl modelo '{nombre_archivo_red}' tiene una perdida final sobre el test: {loss_final}")


    ## Actualizamos la red en su archivo original ##
    if save_after_training:
        if loss_final < init_loss or override_guardado:
            guardar_MLP(NN,nombre_archivo_red)
        else:
            print( f"El error del modelo '{nombre_archivo_red}' sobre el test no ha mejorarado y por tanto NO la actualizamos.\n")
    else:
        print(f"Se ha decidido no guardar la actualizacion del modelo '{nombre_archivo_red}'\n.")

