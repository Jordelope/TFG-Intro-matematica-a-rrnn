import torch
import torch.nn.functional as F
from MLP import MLP, cargar_MLP, guardar_MLP
from Procesamiento_datos_modular import Xs_entrenamiento_def, Xs_test_def, Ys_entrenamiento_def, Ys_test_def
from Autoencoder import Autoencoder, guardar_autoencoder, cargar_autoencoder
from Clasificador import Clasificador, guardar_classificador, cargar_classificador

"""

Crear un Clasificador a partir de un Autoencoder YA ENTRENADO. Clasificamos desde el espacio latente.

"""

## NOMBRE archivos de encoder/Autoencoder y OPCIONES de entrenado y guardado ##
existen_encoder_suelto = True
archivo_encod = r"redes_disponibles\pruebaVisual_dim3_encod.json"
archivo_autoencoder = r"redes_disponibles\pruebaVisual_dim3_autoencod.json" # Si solo tenemos el archivo del autoencoder
archivo_clasificador =  r"redes_disponibles\pruebaClasificador_dimLatente3.json"
train_clasificador = False

save_clasificador = True
save_encoder = False # DEJAR EN False A MENOS QUE NO SE TUVIERA EL ENCODER INDEPENDIENTE


## ESTRUCTURA mlp_clasificador ##

n_classes = 5           # Número de entradas
lat_spc_dim = 3         # Dimension espacio latente(salida encoder, entrada del Clasificador)

estructura_mlp_clasificador= [36, 18 , 10]      # Capas ocultas mlp_clasificador 


f_out_mlp_clas = None                        # Función de activación de salida  decoder    (None = por defecto lineal en MLP)
f_oculta_mlp_clas = F.softmax               # Función de activación capas ocultas decoder (None = por defecto lineal en MLP)




## HIPERPARAMETROS de entrenamiento ##

stp_n = 300                # Número de pasos de entrenamiento
stp_sz = stp_sz = 0.001    # Tamaño del paso (learning rate)
batch_sz = None               # Tamaño del batch (por defecto, todo el dataset)

loss_f = F.cross_entropy   # Función de pérdida


## DATOS de entrenamiento ##
xs = Xs_entrenamiento_def
ys = Ys_entrenamiento_def

#-------------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    
    if not save_clasificador :
        print("AVISO: No se guardará el Clasificador.")
    elif save_encoder:
        print("AVISO: Si existe un fichero para el encoder se va a modificar. Si no existia se va a crear.")
    if train_clasificador:
        print("AVISO: El Clasificador se entrenará.")
    else:
        print("AVISO: El Clasificador no se entrenará")
    

    # Cargamos el encoder (suelto o desde el autoencoder)
    if existen_encoder_suelto:
        encoder = cargar_MLP(archivo_encod)
    else:
        autoencoder = cargar_autoencoder(archivo_autoencoder)
        encoder = autoencoder.encoder
        
    
    
    # Crear Clasificador
    clasificador = Clasificador(encoder,n_classes,estructura_mlp_clasificador,f_out_mlp_clas,f_oculta_mlp_clas)
    

    if train_clasificador:
        if torch.isnan(Xs_entrenamiento_def).any():
            print("[ERROR] Xs_entrenamiento_def contiene NaNs")
        elif torch.isinf(Xs_entrenamiento_def).any():
            print("[ERROR] Xs_entrenamiento_def contiene infinitos")
        else:
            print(f"Iniciamos entrenamiento de {stp_n} pasos del autoencoder '{archivo_autoencoder}'.\n")
            clasificador.train_classifier(xs,ys,stp_n,stp_sz,loss_f,batch_sz)
    

    # Guardamos las redes segun eleccion
    if save_clasificador:
        guardar_classificador(clasificador, archivo_clasificador)
    if save_encoder :
        guardar_MLP(encoder, archivo_encod)
     
