import torch
import torch.nn.functional as F
from MLP import MLP
from Autoencoder import Autoencoder 
from Guardar_Cargar import guardar_modelo, cargar_modelo
from Procesar_datos import procesar_datos



## Funciones relevantes ##



## DATOS de red a entrenar ##

archivo_encod = r"redes_disponibles\encod_nba1.json" 
archivo_decod = r"redes_disponibles\decod_nba1.json" 
archivo_autoencoder =r"redes_disponibles\autoencoder_nba2_mejoras.json" 

## HIPERPARAMETROS de entrenamiento ##

stp_n = 30000     # Número de pasos de entrenamiento
stp_sz = 0.0025   # Tamaño del paso (learning rate)
batch_sz = 32  # Tamaño del batch (por defecto si es None, todo el dataset)

loss_f = F.mse_loss # Función de pérdida
beta = 1e-5
lambda_l2 = 1e-5

## OPCIONES de guardado ##

save_after_training = True  # En caso de True: se guarda cuando mejora el error respecto 
override_guardado = False   # En caso de True: se guarda aunque no mejore el error (si el anterior es True)

sobreescribir_submodelos = False # En caso de True: Se sobreescriben archivos de encoder y decoder.

descripcion = f" Entrenamiento de {stp_n} pasos de tamano {stp_sz} con funcion de perdida {loss_f.__name__} en batches de {batch_sz} y valores beta={beta}, lambda_l2={lambda_l2}.\n Establecemos umbrales de 8mpg y 20pj."
añadir_descripcion = True # Añade a la descripcion ya existente
sustituir_desc = False    # CUIDADO, SI TRUE ELIMINA LA DESCRIPCIÓN YA EXISTENTE
añadir_info_mejora = True # Añade informacion de como ha mejorado/empeorado el modelo sobre el test dado


archivo_entrenamiento = r"datasets\nba\combined19_25_pergame_filtered.csv"
archivo_test = r"datasets\equipos\roster_hawks_pergame_25.csv" # Seria ideal poner de test datos que no hubiera visto
xs_train, ys_train, etiquetas_train, xs_test, ys_test, etiquetas_test = procesar_datos(archivo_set_train=archivo_entrenamiento,
                                                                                    archivo_set_test=archivo_test,
                                                                                    modo_autoencoder=True,
                                                                                    modo_columnas="solo_volumen",
                                                                                    modo_targets="pos",
                                                                                    modo_etiquetado="posicion",
                                                                                    normalizar_datos=True,
                                                                                    modo_normalizacion="zscore",
                                                                                    umbral_partidos=20,
                                                                                    umbral_minutos=8,
                                                                                    umbral_en_test=True,
                                                                                    hay_fila_total_entrenamiento=False,
                                                                                    hay_fila_total_test=True)




#------------------------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    
    ## CARGAR red ##
    print(f"\nSe va entrenar el modelo '{archivo_autoencoder}'.")
    NN = cargar_modelo(archivo_autoencoder)

    ## AVISOS ##
    if  save_after_training:
        if override_guardado:
            print(f"\nAVISO: El modelo '{archivo_autoencoder}' se va a guardar aunque empeore el error.")
        else:
            print(f"\nAVISO: El modelo '{archivo_autoencoder}' se va a guardar.")
        if sobreescribir_submodelos:
            print(f"\nAVISO: Se van a sobrescribir el encoder y decoder.")
        if sustituir_desc:
            print(f"\nAVISO: Se va a ELIMINAR la descripción existente y sustituir por la indicada.")
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

        if añadir_descripcion:
            if añadir_info_mejora:
                descripcion += f"\n El modelo ha mejorado de {loss_init} a {loss_final} sobre el test {archivo_test} tras entrenar con el dataset {archivo_entrenamiento}."
            if sustituir_desc:
                NN.description = descripcion
            else:
                NN.add_descript(descripcion)

        if loss_final < loss_init or override_guardado:
            
            print( f"El error del modelo '{archivo_autoencoder}' sobre el test ha mejorarado y por tanto la actualizamos.\n")
            guardar_modelo(NN,archivo_autoencoder)
            
            if sobreescribir_submodelos:
                guardar_modelo(NN.encoder,archivo_encod)
                guardar_modelo(NN.decoder,archivo_decod)
        else:
            print( f"El error del modelo '{archivo_autoencoder}' sobre el test no ha mejorarado y por tanto NO la actualizamos.\n")
    else:
        print(f"Se ha decidido NO guardar la actualizacion del modelo '{archivo_autoencoder}.'\n")
