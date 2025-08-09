import torch
import torch.nn.functional as F
from MLP import MLP, cargar_MLP, guardar_MLP
from Procesamiento_datos_modular import Xs_entrenamiento_def, Xs_test_def, Ys_entrenamiento_def, Ys_test_def
from Autoencoder import Autoencoder, guardar_autoencoder, cargar_autoencoder
from Clasificador import Clasificador, guardar_classificador, cargar_classificador

"""
(PENDIENTE REVISION Y PRUEBA)

Entrenar Clasificador. " opciones":
-> El clasificador suelto (entrenamiento "habitual")
-> El clasificador entero con el encoder (fine_tuning)

Igual seria util crear archivo con datos ya procesados para no procesarlos cada vez.

Va siendo momento de mejorar el procesamiento de datos para decidir cuales son los target(el procresamiento de los que se reciben esta bien)
"""

## Funciones relevantes ##



## DATOS de red a entrenar ##

nombre_archivo_clasificador = r"redes_disponibles\pruebaClasificador_dimLatente3.json"
nombre_archivo_encoder = r"redes_disponibles\pruebaVisual_dim3_encod.json"

## OPCIONES de guardado ##

modo_fine_tunning = False   # Por lo general dejar en false 
save_after_training = True  # En caso de True: se guarda cuando mejora el error respecto 
override_guardado = True   # En caso de True: se guarda aunque no mejore el error (si el anterior es True)


## HIPERPARAMETROS de entrenamiento ##

stp_n = 10000     # Número de pasos de entrenamiento
stp_sz = 0.0005    # Tamaño del paso (learning rate)
batch_sz = None  # Tamaño del batch (por defecto si es None, todo el dataset)

loss_f = F.cross_entropy # Función de pérdida


## DATOS de entrenamiento y test (sin encoding)##
xs_train = Xs_entrenamiento_def
ys_train = Ys_entrenamiento_def

xs_test = Xs_test_def
ys_test = Ys_test_def



#------------------------------------------------------------------------------------------------------------------------------------



if __name__ == "__main__":
    
    ## CARGAR red ##
    print(f"\nSe va entrenar el modelo '{nombre_archivo_clasificador}'.")
    NN = cargar_classificador(nombre_archivo_clasificador)
    encoder = NN.encoder


    ## AVISOS ##
    if  save_after_training:
        if override_guardado:
            print(f"\nAVISO: El modelo '{nombre_archivo_clasificador}' se va a guardar aunque empeore el error.")
        else:
            print(f"\nAVISO: El modelo '{nombre_archivo_clasificador}' se va a guardar.")
        if modo_fine_tunning:
            print(f"\nAVISO: MODO FINE-TUNING ACTIVADO. Los parametros del encoder '{nombre_archivo_clasificador}' van a ser modificados.")
            
    else:
        print(f"\nAVISO: El modelo '{nombre_archivo_clasificador}' no se va a guardar.")

    
    ## CODIFICAMOS DATOS si solo entrenamos clasificador (NN(x identifica bien, pero la funcion train los requiere codificados)) ##
    if not modo_fine_tunning:
        xs_train_encoded = encoder(xs_train)
        xs_test_encoded =  encoder(xs_test)


    ## ERROR INICIAL sobre el test ##
    with torch.no_grad():
        pred_test_init = [NN(x) for x in xs_test_encoded]
        loss_init = sum( loss_f(yout, ytrue) for yout,ytrue in zip(pred_test_init, ys_test) ) / len(ys_test) 
    print(f"\nEl modelo '{nombre_archivo_clasificador}' tiene una perdida inicial sobre el test: {loss_init}")


    ## ENTRENAMIENTO ##
    print(f"\nIniciamos entrenamiento de {stp_n} pasos de la red '{nombre_archivo_clasificador}'.\n") 
    if modo_fine_tunning:
        NN.train_whole_model(xs_train,ys_train,stp_n,stp_sz,loss_f,batch_sz) #Aun no esta operativo
    else:
        NN.train_classifier(xs_train_encoded,ys_train,stp_n,stp_sz,loss_f,batch_sz)


    ## ERROR FINAL sobre el test ##
    with torch.no_grad():
        pred_test_fin = [NN(x) for x in xs_test_encoded]
        loss_final = sum( loss_f(yout, ytrue) for yout,ytrue in zip(pred_test_fin, ys_test) ) / len(ys_test) 
    print(f"\nLa red '{nombre_archivo_clasificador}' tiene una perdida final sobre el test: {loss_final}")

    
    ## GUARDADO de la red en su archivo original ##
    if save_after_training:
        if loss_final < loss_init or override_guardado:
            print( f"El error de la red '{nombre_archivo_clasificador}' sobre el test ha mejorarado y por tanto la actualizamos.\n")
            guardar_classificador(NN,nombre_archivo_clasificador)
            if modo_fine_tunning:
                print(f"Se ha entrenado en modo fine-tunning y por tanto actualizamos el encoder.")
                guardar_MLP(encoder,nombre_archivo_encoder)

        else:
            print( f"El error de la red '{nombre_archivo_clasificador}' sobre el test no ha mejorarado y por tanto NO la actualizamos.\n")
    else:
        print(f"Se ha decidido NO guardar la actualizacion de la red '{nombre_archivo_clasificador}'\n.")
