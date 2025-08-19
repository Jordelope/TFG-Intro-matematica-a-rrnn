import torch
import torch.nn.functional as F
from MLP import MLP 
from Autoencoder import Autoencoder 
from Clasificador import Clasificador 
from Guardar_Cargar import guardar_modelo, cargar_modelo
from Procesar_datos import procesar_datos

"""

PENDIENTE:
    -FAlla al entrenar -> revisar encaje cross entropy softmax (el fallo parece estar en el bucle entrenamiento de mlp)


"""
## Funciones relevantes ##
def onehot_to_long(targets: torch.Tensor) -> torch.Tensor:
    """
    Para cuando se use F.cross_entropy.
    Detecta si los targets están en formato one-hot y los convierte a índices de clase (long).
    Si ya están en formato entero, los deja tal cual.
    """
    # Verifica si es un tensor 2D y si cada fila tiene una única posición con valor 1
    if targets.ndim == 2 and torch.all((targets.sum(dim=1) == 1)) and torch.all((targets == 0) | (targets == 1)):
        # Es one-hot → convertir a índices
        return torch.argmax(targets, dim=1).long()
    else:
        # Ya está en formato correcto o no es one-hot
        return targets.long()


## NOMBRE archivos de encoder/Autoencoder y clasificador ##
existe_encoder_suelto = False
existe_mlp_clas_suelto = False

archivo_encod =         r"redes_disponibles\encod_prueba_desc.json" 
archivo_autoencoder =   r"redes_disponibles\autoen_prueba_desc.json"  # Si solo tenemos el archivo del autoencoder
archivo_mlp_clas =      r"redes_disponibles\mlp_clas_prueba_desc.json" 
archivo_clasificador =  r"redes_disponibles\clasificador_prueba_desc.json" 

## OPCIONES de entrenado y guardado ##
save_clasificador = True
save_mlp_clas = True   # RECOMENDACION: Dejar en False
save_encoder = False    # RECOMENDADCION: Dejar en False a menos que NO se tenga encoder independiente.

train_clasificador = False


añadir_descripcion = False  # Opción de añadir una descripción
descripcion = ""


## ESTRUCTURA mlp_clasificador ##
n_classes = 5                            # Número de clases de nuestro clasificador
estructura_oc_mlp_clasificador= [8]      # Capas ocultas mlp_clasificador 

lista_activaciones_mlp_clas = [torch.relu for i in range(len(estructura_oc_mlp_clasificador))] + [F.softmax] # Funciones activacion del mlp_clas


## HIPERPARAMETROS de entrenamiento ##

stp_n = 1050             # Número de pasos de entrenamiento
stp_sz = 0.001           # Tamaño del paso (learning rate)
batch_sz = None          # Tamaño del batch (por defecto, todo el dataset)

loss_f = F.cross_entropy   # Función de pérdida


## DATOS de entrenamiento y test ##
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
                                                                                    umbral_partidos=20,
                                                                                    umbral_minutos=15,
                                                                                    umbral_en_test=True,
                                                                                    hay_fila_total_entrenamiento=False,
                                                                                    hay_fila_total_test=True)


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
        encoder = cargar_modelo(archivo_encod)
    else:
        autoencoder = cargar_modelo(archivo_autoencoder)
        encoder = autoencoder.encoder
    
    # Crear Clasificador
    clasificador = Clasificador(encoder,n_classes,estructura_oc_mlp_clasificador,lista_activaciones_mlp_clas)
    if existe_mlp_clas_suelto:
        mlp_clasificador = cargar_modelo(archivo_mlp_clas)
        clasificador.mlp_clasificador = mlp_clasificador


    if train_clasificador:
        if torch.isnan(xs_train).any():
            print("[ERROR] Xs_entrenamiento_def contiene NaNs")
        elif torch.isinf(xs_train).any():
            print("[ERROR] Xs_entrenamiento_def contiene infinitos")
        else:
            
            # Si usamos cross entropy procesamos los targets
            if loss_f == F.cross_entropy:
                ys_train = onehot_to_long(ys_train)
                ys_test = onehot_to_long(ys_test)

            ## ERROR INICIAL sobre el test ##
            with torch.no_grad():
                pred_test_init = clasificador(xs_test)
                loss_init = loss_f(pred_test_init,ys_test)
            print(f"\nEl modelo '{archivo_clasificador}' tiene una perdida inicial sobre el test: {loss_init}.\n")

            ## ENTRENAMIENTO de clasificador##
            print(f"\nIniciamos entrenamiento de {stp_n} pasos de la red '{archivo_clasificador}'.\n")
            clasificador.train_classifier(xs_train,ys_train,stp_n,stp_sz,loss_f,batch_sz)
            
            ## ERROR FINAL sobre el test ##
            with torch.no_grad():
                pred_test_fin = clasificador(xs_test)
                loss_final = loss_f(pred_test_fin,ys_test)
            print(f"\nEl modelo '{archivo_clasificador}' tiene una perdida final sobre el test: {loss_final}.\n")


    # Guardamos las redes segun eleccion
    if añadir_descripcion:
        clasificador.description = descripcion
    if save_clasificador:
        guardar_modelo(clasificador, archivo_clasificador)
        if save_encoder :
            guardar_modelo(encoder, archivo_encod)
        if save_mlp_clas:
            guardar_modelo(clasificador.mlp_clasificador,archivo_mlp_clas)
     
