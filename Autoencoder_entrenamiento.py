import torch
import torch.nn.functional as F
from Autoencoder import Autoencoder, cargar_autoencoder, guardar_autoencoder
from Procesamiento_datos_modular import Xs_entrenamiento_def, Ys_entrenamiento_def, Xs_test_def, Ys_test_def


## Funciones relevantes ##



## DATOS de red a entrenar ##

nombre_archivo_red = r"redes_disponibles\pruebaVisual_dim3_autoencod.json"


## OPCIONES de guardado ##

save_after_training = True  # En caso de True: se guarda cuando mejora el error respecto 
override_guardado = True   # En caso de True: se guarda aunque no mejore el error (si el anterior es True)


## HIPERPARAMETROS de entrenamiento ##

stp_n = 10000     # Número de pasos de entrenamiento
stp_sz = 0.0005    # Tamaño del paso (learning rate)
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
    print(f"\nSe va entrenar el modelo '{nombre_archivo_red}'.")
    NN = cargar_autoencoder(nombre_archivo_red)

    ## AVISOS ##
    if  save_after_training:
        if override_guardado:
            print(f"\nAVISO: La red '{nombre_archivo_red}' se va a guardar aunque empeore el error.")
        else:
            print(f"\nAVISO: La red '{nombre_archivo_red}' se va a guardar.")
    else:
        print(f"\nAVISO: La red '{nombre_archivo_red}' no se va a guardar.")


    ## ERROR INICIAL sobre el test ##
    with torch.no_grad():
        pred_test_init = [NN(x) for x in xs_test]
        loss_init = sum( loss_f(yout, ytrue) for yout,ytrue in zip(pred_test_init, ys_test) ) / len(ys_test) 
    print(f"\nLa red '{nombre_archivo_red}' tiene una perdida inicial sobre el test: {loss_init}")


    ## ENTRENAMIENTO ##
    print(f"\nIniciamos entrenamiento de {stp_n} pasos de la red '{nombre_archivo_red}'.\n") 
    NN.train_model(xs_train,stp_n,stp_sz,loss_f,batch_sz)


    ## ERROR FINAL sobre el test ##
    with torch.no_grad():
        pred_test_fin = [NN(x) for x in xs_test]
        loss_final = sum( loss_f(yout, ytrue) for yout,ytrue in zip(pred_test_fin, ys_test) ) / len(ys_test) 
    print(f"\nLa red '{nombre_archivo_red}' tiene una perdida final sobre el test: {loss_final}")

    
    ## GUARDADO de la red en su archivo original ##
    if save_after_training:
        if loss_final < loss_init or override_guardado:
            print( f"El error de la red '{nombre_archivo_red}' sobre el test ha mejorarado y por tanto la actualizamos.\n")
            guardar_autoencoder(NN,nombre_archivo_red)
        else:
            print( f"El error de la red '{nombre_archivo_red}' sobre el test no ha mejorarado y por tanto NO la actualizamos.\n")
    else:
        print(f"Se ha decidido NO guardar la actualizacion de la red '{nombre_archivo_red}'\n.")
