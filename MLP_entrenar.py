import torch
import torch.nn.functional as F
import random
from MLP import MLP
from Guardar_Cargar import cargar_modelo, guardar_modelo
from Procesar_datos import procesar_datos

## Funciones relevantes ##
def clasificacion(xs):
    return xs.argmax().item()  # Devuelve el índice de la clase con mayor probabilidad


## MODELO a entrenar ##

nombre_archivo_red = r"redes_disponibles\mlp_prueba_desc.json"  # Archivo donde se guarda la red


## HIPERPARAMETROS de entrenamiento ##
stp_n = 5     # Número de pasos de entrenamiento
stp_sz = 0.001    # Tamaño del paso (learning rate)
batch_sz = None  # Tamaño del batch (por defecto, todo el dataset)

loss_f = F.mse_loss # Funcion de perdida


## OPCIONES de guardado ##  
save_after_training = True  # En caso de True: se guarda cuando mejora el error respecto 
override_guardado = True   # En caso de True: se guarda aunque no mejore el error (si el anterior es True)

descripcion = f"Entrenamiento de {stp_n} pasos de tamano {stp_sz} con funcion de perdida {loss_f.__name__} en batches de {batch_sz}."
añadir_descripcion = True
sustituir_desc = False # CUIDADO, ELIMINA LA DESCRIPCIÓN ANTERIOR
añadir_info_mejora = True # Añade informacion de como ha mejorado/empeorado el modelo sobre el test dado

## DATOS de test y entrenamiento ##
archivo_entrenamiento = "datasets/nba_pergame_24_full.csv"
archivo_test = "datasets/nba_pergame_24_full.csv"
xs_train, ys_train, etiquetas_train, xs_test, ys_test, etiquetas_test = procesar_datos(archivo_set_train=archivo_entrenamiento,
                                                                                    archivo_set_test=archivo_test,
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
    NN = cargar_modelo(nombre_archivo_red)


    ## AVISOS ##
    if  save_after_training:
        if override_guardado:
            print(f"\nAVISO: El modelo '{nombre_archivo_red}' se va a guardar aunque empeore el error.")
        else:
            print(f"\nAVISO: El modelo '{nombre_archivo_red}' se va a guardar.")
        
        if sustituir_desc:
             print(f"\nAVISO: Se va a ELIMINAR la descripción existente y sustituir por la indicada.")

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
        if añadir_descripcion:
            if añadir_info_mejora:
                descripcion += f"\n El modelo ha mejorado de {init_loss} a {loss_final} sobre el test {archivo_test} tras entrenar con el dataset {archivo_entrenamiento}."
            if sustituir_desc:
                NN.description = descripcion
            else:
                NN.add_descript(descripcion)


        if loss_final < init_loss or override_guardado:
            guardar_modelo(NN,nombre_archivo_red)
        else:
            print( f"El error del modelo '{nombre_archivo_red}' sobre el test no ha mejorarado y por tanto NO la actualizamos.\n")
    else:
        print(f"Se ha decidido no guardar la actualizacion del modelo '{nombre_archivo_red}'\n.")

