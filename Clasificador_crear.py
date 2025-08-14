import torch
import torch.nn.functional as F
from MLP_mejoras import MLP, cargar_MLP, guardar_MLP
from Procesamiento_datos_modular import Xs_entrenamiento_def, Xs_test_def, Ys_entrenamiento_def, Ys_test_def
from Autoencoder_mejoras import Autoencoder, guardar_autoencoder, cargar_autoencoder
from Clasificador import Clasificador, guardar_classificador, cargar_classificador

"""

PENDIENTE:
    -FAlla al entrenar -> revisar encaje cross entropy softmax (el fallo parece estar en el bucle entrenamiento de mlp)

Crear un Clasificador a partir de un Autoencoder YA ENTRENADO. Clasificamos desde el espacio latente.

"""

## NOMBRE archivos de encoder/Autoencoder y OPCIONES de entrenado y guardado ##
existe_encoder_suelto = False
existe_mlp_clas_suelto = False

archivo_encod =         r"redes_disponibles\mejoras\encoder_pruebas.json"
archivo_autoencoder =   r"redes_disponibles\mejoras\autoencoder_pruebas.json" # Si solo tenemos el archivo del autoencoder
archivo_mlp_clas =      r"redes_disponibles\mejoras\mlp_clas_pruebas.json"
archivo_clasificador =  r"redes_disponibles\mejoras\clasificador_pruebas.json"

train_clasificador = True

save_clasificador = True
save_mlp_clas = False   # RECOMENDACION: Dejar siempre en False
save_encoder = False    # RECOMENDADCION: Dejar en False a menos que NO se tenga encoder independiente.


## ESTRUCTURA mlp_clasificador ##
n_classes = 5           # Número de clases de nuestro clasificador

estructura_oc_mlp_clasificador= [36, 18 , 10]      # Capas ocultas mlp_clasificador 

lista_activaciones_mlp_clas = [None for i in range(len(estructura_oc_mlp_clasificador))] + [F.softmax] # Funciones activacion del mlp_clas


## HIPERPARAMETROS de entrenamiento ##

stp_n = 1                # Número de pasos de entrenamiento
stp_sz = 0.001           # Tamaño del paso (learning rate)
batch_sz = None          # Tamaño del batch (por defecto, todo el dataset)

loss_f = F.cross_entropy   # Función de pérdida


## DATOS de entrenamiento ##
xs_train = Xs_entrenamiento_def
ys_train = Ys_entrenamiento_def
xs_test = Xs_test_def
ys_test = Ys_test_def

#-------------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    
    if not save_clasificador :
        print("AVISO: No se guardará el Clasificador.\n")
    else:
        if save_encoder:
            print("AVISO: Si existe un fichero para el encoder se va a modificar. Si no existia se va a crear.\n")
        if save_mlp_clas:
            print("AVISO: Si existe un fichero para el mlp_clas se va a modificar. Si no existia se va a crear.\n")

    if train_clasificador:
        print("AVISO: El Clasificador se entrenará.\n")
    else:
        print("AVISO: El Clasificador no se entrenará.\n")
    

    # Cargamos el encoder (suelto o desde el autoencoder)
    if existe_encoder_suelto:
        encoder = cargar_MLP(archivo_encod)
    else:
        autoencoder = cargar_autoencoder(archivo_autoencoder)
        encoder = autoencoder.encoder
    
    # Crear Clasificador
    clasificador = Clasificador(encoder,n_classes,estructura_oc_mlp_clasificador,lista_activaciones_mlp_clas)
    if existe_mlp_clas_suelto:
        mlp_clasificador = cargar_MLP(archivo_mlp_clas)
        clasificador.mlp_clasificador = mlp_clasificador


    if train_clasificador:
        if torch.isnan(Xs_entrenamiento_def).any():
            print("[ERROR] Xs_entrenamiento_def contiene NaNs")
        elif torch.isinf(Xs_entrenamiento_def).any():
            print("[ERROR] Xs_entrenamiento_def contiene infinitos")
        else:
            
            ## ERROR INICIAL sobre el test ##
            with torch.no_grad():
                pred_test_init = clasificador(xs_test)
                loss_init = loss_f(pred_test_init,ys_test)
            print(f"\nEl modelo '{archivo_clasificador}' tiene una perdida inicial sobre el test: {loss_init}.\n")

            ## ENTRENAMIENTO ##
            print(f"\nIniciamos entrenamiento de {stp_n} pasos de la red '{archivo_clasificador}'.\n")
            clasificador.train_classifier(xs_train,ys_train,stp_n,stp_sz,loss_f,batch_sz)
            
            ## ERROR FINAL sobre el test ##
            with torch.no_grad():
                pred_test_fin = clasificador(xs_test)
                loss_final = loss_f(pred_test_fin,ys_test)
            print(f"\nEl modelo '{archivo_clasificador}' tiene una perdida final sobre el test: {loss_final}.\n")


    # Guardamos las redes segun eleccion
    if save_clasificador:
        guardar_classificador(clasificador, archivo_clasificador)
        if save_encoder :
            guardar_MLP(encoder, archivo_encod)
        if save_mlp_clas:
            guardar_MLP(clasificador.mlp_clasificador,archivo_mlp_clas)
     
