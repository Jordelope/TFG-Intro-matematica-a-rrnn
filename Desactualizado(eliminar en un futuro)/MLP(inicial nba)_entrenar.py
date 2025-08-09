import torch
import torch.nn.functional as F
import random
from MLP import MLP, guardar_MLP, cargar_MLP
from proc_datos_entrenamiento import Xs_entrenamiento_def, Xs_test_def, Ys_entrenamiento_def, Ys_test_def


## Funciones relevantes ##
def decodificar_pos(xs):
    posiciones = ["PG","SG","SF","PF","C"]
    if xs is None:
        return "UNKNOWN"

    return posiciones[xs.argmax().item()]


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
override_guardado = True   # En caso de True: se guarda aunque no mejore el error (si el anterior es True)


## Datos de test y entrenamiento ##
Xs_train = Xs_entrenamiento_def
Ys_train = Ys_entrenamiento_def

Xs_test = Xs_test_def
Ys_test = Ys_test_def
pos_test = [decodificar_pos(y) for y in Ys_test] # Las posiciones reales


#-------------------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":

    print(f"\nSe va entrenar el modelo '{nombre_archivo_red}'.")
    NN = cargar_MLP(nombre_archivo_red)


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
    test_pred_fin_pos =  [decodificar_pos(x) for x in test_pred_fin]
    
    cont_buenos_final = 0
    for i in range(len(test_pred_fin_pos)):
        if test_pred_fin_pos[i] == pos_test[i]: 
            cont_buenos_final +=1
    accuracy_fin = cont_buenos_final / len(pos_test)

    print(f"\nLa red sobre el test dice: \n{test_pred_fin_pos}")
    print(f"Deberia dar: \n{pos_test}")
    print(f"La red ha acertado {cont_buenos_final} / {len(test_pred_fin)} despues del entrenamiento.")
    print(f"Precisión final en test: {accuracy_fin*100:.2f}%\n")


    ## Actualizamos la red en su archivo original ##
    if save_after_training:
        if loss_final < init_loss or override_guardado:
            guardar_MLP(NN,nombre_archivo_red)
        else:
            print( f"El error de la red '{nombre_archivo_red}' sobre el test no ha mejorarado y por tanto no la actualizamos.\n")
    else:
        print(f"Se ha decidido no guardar la actualizacion de la red '{nombre_archivo_red}'\n.")

