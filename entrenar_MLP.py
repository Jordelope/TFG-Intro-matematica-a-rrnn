import torch
import torch.nn.functional as F
import random
from MLP import MLP, guardar_red, cargar_red
from procesar_datos_entrenamiento import Xs_entrenamiento_def, Xs_test_def, Ys_entrenamiento_def, Ys_test_def


## Funciones relevantes ##
def clasificacion(xs):
    return xs.argmax().item()  # Devuelve el índice de la clase con mayor probabilidad


## Establecemos parametros entrenamiento ##
xs = Xs_entrenamiento_def
ys = Ys_entrenamiento_def

stp_n = 21
stp_sz = 0.05
batch_sz = len(xs)

loss_f = F.cross_entropy


## Datos de red a entrenar ##
nombre_archivo_red = r"redes_disponibles/red_prueba_med.json"

# Decidir si se quiere modificar el archivo  despues de entrenar 
save_after_training = True  # En caso de True: se guarda cuando mejora el error respecto 
override_guardado = False   # En caso de True: se guarda aunque no mejore el error (si el anterior es True)


## Datos de test y entrenamiento ##
Xs_train = Xs_entrenamiento_def
Ys_train = Ys_entrenamiento_def

Xs_test = Xs_test_def
Ys_test = Ys_test_def



#-------------------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":

    print(f"\nSe va entrenar el modelo '{nombre_archivo_red}'.")
    NN = cargar_red(nombre_archivo_red)


    if  save_after_training:
        if override_guardado:
            print(f"\nAVISO: La red '{nombre_archivo_red}' se va a guardar aunque empeore el error.")
        else:
            print(f"\nAVISO: La red '{nombre_archivo_red}' se va a guardar.")
    else:
        print(f"\nAVISO: La red '{nombre_archivo_red}' no se va a guardar.")


    ## Calculamos error inicial sobre training set para analizar la mejora ##
    pred_entrenamiento_init = [NN(x) for x in Xs_train]
    init_loss = sum( F.cross_entropy(yout, ytrue) for yout,ytrue in zip(pred_entrenamiento_init, Ys_train) ) / len(Ys_train) 
    print(f"\nLa red '{nombre_archivo_red}' tiene una perdida inicial sobre el set de entrenamiento: {init_loss}")

    ## Procedemos a entrenar ##
    print(f"\nIniciamos entrenamiento de {stp_n} pasos de la red '{nombre_archivo_red}'.\n") 
    NN.train_model(Xs_train,Ys_train,stp_n,stp_sz,F.cross_entropy,batch_sz)
    
    
    ## Calculamos error final sobre training set ##
    pred_entrenamiento_fin = [NN(x) for x in Xs_train]
    loss_final = sum( F.cross_entropy(yout, ytrue) for yout,ytrue in zip(pred_entrenamiento_fin, Ys_train) ) / len(Ys_train) 
    print(f"\nLa red '{nombre_archivo_red}' tiene una perdida final sobre el set de entrenamiento: {loss_final}")


    ## Analizamos el rendimiento tras el entrenamiento con el test ##
    test_pred_fin = [ NN(x) for x in Xs_test]
    class_test = [clasificacion(x) for x in test_pred_fin]  # Decodificamos las clases predichas
    class_pred = [clasificacion(y) for y in Ys_test]  # Decodificamos las clases reales
    
    cont_buenos_final = 0
    for i in range(len(class_test)):
        if class_pred[i] == class_test[i]: 
            cont_buenos_final +=1
    accuracy_fin = cont_buenos_final / len(class_pred)

    print(f"\nLa red sobre el test dice: \n{class_pred}")
    print(f"Deberia dar: \n{class_test}")
    print(f"La red ha acertado {cont_buenos_final} / {len(class_pred)} despues del entrenamiento.")
    print(f"Precisión final en test: {accuracy_fin*100:.2f}%\n")


    ## Actualizamos la red en su archivo original ##
    if save_after_training:
        if loss_final < init_loss or override_guardado:
            guardar_red(NN,nombre_archivo_red)
        else:
            print( f"El error de la red '{nombre_archivo_red}' sobre el test no ha mejorarado y por tanto no la actualizamos.\n")
    else:
        print(f"Se ha decidido no guardar la actualizacion de la red '{nombre_archivo_red}'\n.")

