import torch
import torch.nn.functional as F
from MLP import MLP, cargar_MLP, guardar_MLP
from Procesar_datos import procesar_datos
from Autoencoder import Autoencoder, guardar_autoencoder, cargar_autoencoder
from Clasificador import Clasificador, guardar_classificador, cargar_classificador

"""
(PENDIENTE REVISION Y PRUEBA)

Entrenar Clasificador. " opciones":
-> El clasificador suelto (entrenamiento "habitual")
-> El clasificador entero con el encoder (fine_tuning)

Igual seria util crear archivo con datos ya procesados para no procesarlos cada vez.
"""




## DATOS de red a entrenar ##

archivo_clasificador = r"redes_disponibles\prueba_crossEntropy.json"
archivo_mlp_clas =  r"redes_disponibles\mlp_clas_pruebas.json"

## OPCIONES de guardado  ##

save_after_training = True  # En caso de True: se guarda cuando mejora el error respecto 
override_guardado = False   # En caso de True: se guarda aunque no mejore el error (si el anterior es True)
save_mlp_clas = False


## HIPERPARAMETROS de entrenamiento ##

stp_n = 1000    # Número de pasos de entrenamiento
stp_sz = 0.000005   # Tamaño del paso (learning rate)
batch_sz = None  # Tamaño del batch (por defecto si es None, todo el dataset)

loss_f = F.cross_entropy # Función de pérdida


## DATOS de entrenamiento y test (sin encoding)##
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
    print(f"\nSe va entrenar el modelo '{archivo_clasificador}'.\n")
    NN = cargar_classificador(archivo_clasificador)
    encoder = NN.encoder


    ## AVISOS ##
    if  save_after_training:
        if override_guardado:
            print(f"\nAVISO: El modelo '{archivo_clasificador}' se va a guardar aunque empeore el error.")
        else:
            print(f"\nAVISO: El modelo '{archivo_clasificador}' se va a guardar.")
        if save_mlp_clas:
            print(f"\nAVISO: Se va a guardar el mlp_clasificador.")
            
    else:
        print(f"\nAVISO: El modelo '{archivo_clasificador}' no se va a guardar.")

    # Si usamos cross entropy procesamos los targets
    #if loss_f == F.cross_entropy:
    #    ys_train = onehot_to_long(ys_train)
    #    ys_test = onehot_to_long(ys_test)

    ## ERROR INICIAL sobre el test ##
    with torch.no_grad():
        pred_test_init = NN(xs_test)
        loss_init = loss_f(pred_test_init, ys_test)
    print(f"\nEl modelo '{archivo_clasificador}' tiene una perdida inicial sobre el test: {loss_init}.\n")


    ## ENTRENAMIENTO ##
    print(f"\nIniciamos entrenamiento de {stp_n} pasos del modelo '{archivo_clasificador}'.\n") 
    NN.train_classifier(xs_train,ys_train,stp_n,stp_sz,loss_f,batch_sz)


    ## ERROR FINAL sobre el test ##
    with torch.no_grad():
        pred_test_fin = NN(xs_test)
        loss_final = loss_f(pred_test_fin, ys_test)
    print(f"\nLa red '{archivo_clasificador}' tiene una perdida final sobre el test: {loss_final}.\n")

    
    ## GUARDADO de la red en su archivo original ##
    if save_after_training:
        if loss_final < loss_init or override_guardado:
            print( f"El error de la red '{archivo_clasificador}' sobre el test ha mejorarado y por tanto la actualizamos.\n")
            guardar_classificador(NN, archivo_clasificador)
            if save_mlp_clas:
                guardar_MLP(encoder,archivo_mlp_clas)

        else:
            print( f"El error de la red '{archivo_clasificador}' sobre el test no ha mejorarado y por tanto NO la actualizamos.\n")
    else:
        print(f"Se ha decidido NO guardar la actualizacion de la red '{archivo_clasificador}'\n.")
