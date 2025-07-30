import torch
import torch.nn.functional as F
import random
from MLP import MLP, guardar_MLP, cargar_MLP
from proc_datos_entrenamiento import Xs_entrenamiento_def, Xs_test_def, Ys_entrenamiento_def, Ys_test_def


## Funciones relevantes ##
def clasificacion(xs):
    return xs.argmax().item()  # Devuelve el índice de la clase con mayor probabilidad


## Datos de red a entrenar ##

nombre_archivo_red = r"redes_disponibles/red_prueba_med.json"


## HIPERPARAMETROS de entrenamiento ##

stp_n = 300     # Número de pasos de entrenamiento
stp_sz = stp_sz = 0.001    # Tamaño del paso (learning rate)
batch_sz = 8  # Tamaño del batch (por defecto, todo el dataset)

loss_f = F.cross_entropy # Funcion de perdida


## OPCIONES de guardado ##  
save_after_training = True  # En caso de True: se guarda cuando mejora el error respecto 
override_guardado = False   # En caso de True: se guarda aunque no mejore el error (si el anterior es True)


## DATOS de test y entrenamiento ##
Xs_train = Xs_entrenamiento_def
Ys_train = Ys_entrenamiento_def

Xs_test = Xs_test_def
Ys_test = Ys_test_def



#-------------------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    
    ## CARGAR red ##
    print(f"\nSe va entrenar el modelo '{nombre_archivo_red}'.")
    NN = guardar_MLP(nombre_archivo_red)


    ## AVISOS ##
    if  save_after_training:
        if override_guardado:
            print(f"\nAVISO: La red '{nombre_archivo_red}' se va a guardar aunque empeore el error.")
        else:
            print(f"\nAVISO: La red '{nombre_archivo_red}' se va a guardar.")
    else:
        print(f"\nAVISO: La red '{nombre_archivo_red}' NO se va a guardar.")


    ## Calculamos error inicial sobre el test ##
    pred_test_init = [NN(x) for x in Xs_test]
    init_loss = sum( loss_f(yout, ytrue) for yout,ytrue in zip(pred_test_init, Ys_test) ) / len(Ys_test) 
    print(f"\nLa red '{nombre_archivo_red}' tiene una perdida inicial sobre el set de entrenamiento: {init_loss}")

    ## ENTRENAMIENTO ##
    print(f"\nIniciamos entrenamiento de {stp_n} pasos de la red '{nombre_archivo_red}'.\n") 
    NN.train_model(Xs_train,Ys_train,stp_n,stp_sz,loss_f,batch_sz)
    
    
    ## ERROR FINAL sobre el test ##
    pred_test_fin = [NN(x) for x in Xs_test]
    loss_final = sum( F.cross_entropy(yout, ytrue) for yout,ytrue in zip(pred_test_fin, Ys_test) ) / len(Ys_train) 
    print(f"\nLa red '{nombre_archivo_red}' tiene una perdida final sobre el test: {loss_final}")


    ## Actualizamos la red en su archivo original ##
    if save_after_training:
        if loss_final < init_loss or override_guardado:
            guardar_MLP(NN,nombre_archivo_red)
        else:
            print( f"El error de la red '{nombre_archivo_red}' sobre el test no ha mejorarado y por tanto NO la actualizamos.\n")
    else:
        print(f"Se ha decidido no guardar la actualizacion de la red '{nombre_archivo_red}'\n.")

