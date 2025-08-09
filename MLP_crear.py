import torch
import torch.nn.functional as F
import random
from MLP import MLP, guardar_red, cargar_red
from Procesamiento_datos_modular import Xs_entrenamiento_def, Xs_test_def, Ys_entrenamiento_def, Ys_test_def

"""
crear_red_torch.py
-------------------

Este script permite crear, entrenar y guardar una red neuronal multicapa (MLP) utilizando la implementación matemática definida en el módulo MLP.py.
El usuario puede definir la arquitectura de la red, los parámetros de entrenamiento y decidir si desea entrenar y/o guardar la red resultante en un archivo JSON.

Estructura general del script:
- Carga los datos de entrenamiento y test desde procesar_datos_entrenamiento.py.
- Permite definir la arquitectura y funciones de activación de la red MLP.
- Permite entrenar la red con los parámetros elegidos (número de pasos, tamaño de batch, función de pérdida, etc).
- Permite guardar la red (estructura y pesos) en un archivo JSON para su posterior uso.
- Incluye utilidades para decodificar la salida de la red y evaluar su precisión sobre el conjunto de test.

Parámetros principales:
- estructura: lista con el número de neuronas en cada capa oculta.
- f_a_salida, f_a_oculta: funciones de activación para la salida y las capas ocultas.
- stp_n, stp_sz, batch_sz: hiperparámetros de entrenamiento.
- nombre_archivo_red: ruta donde se guarda la red entrenada.
- save_new_NN, entrenar_nueva_red: flags para controlar el flujo del script.

El script está pensado para ser ejecutado como programa principal (__main__), mostrando mensajes informativos sobre las acciones realizadas.
"""
## Funciones relevantes ##


## Cargamos los datos de entrenamiento ##
xs = Xs_entrenamiento_def
ys = Ys_entrenamiento_def


## HIPERPARAMETROS de entrenamiento ##
stp_n = 1000  # Número de pasos de entrenamiento
stp_sz = 0.25 # Tamaño del paso (learning rate)
batch_sz = len(xs)  # Tamaño del batch (por defecto, todo el dataset)

loss_f = F.cross_entropy # Función de pérdida


## ARQUITECTURA de la red ##
input_sz = len(xs[0])      # Número de entradas
out_sz = len(ys[0])        # Número de salidas
estructura = [18, 18]      # Capas ocultas
f_a_salida = None          # Función de activación de salida (None = lineal o definida en MLP)
f_a_oculta = None          # Función de activación oculta (None = por defecto en MLP)
nombre_archivo_red = r"redes_disponibles\patata.json"  # Archivo donde se guarda la red

## OPCIONES GUARDADO ##
save_new_NN = True         # ¿Guardar la red tras crearla?
entrenar_nueva_red = False # ¿Entrenar la red tras crearla?

# Instanciación de la red
NN = MLP(input_sz, out_sz, estructura, f_a_salida, f_a_oculta)


## Establecemos el test ## 
test_xs = Xs_test_def
test_ys = Ys_test_def


#-------------------------------------------------------------------------------------------------------------------------------



if __name__ == "__main__":
    print(f"Hemos creado la red {nombre_archivo_red}.")

    if save_new_NN:
        print(f"\nAVISO: La red {nombre_archivo_red} se va a guardar.")
    else:
        print(f"\nAVISO: La red {nombre_archivo_red} no se va a guardar.")

    if entrenar_nueva_red:
        print(f"\nAVISO: La red {nombre_archivo_red} se va a entrenar.")
    else:
        print(f"\nAVISO: La red {nombre_archivo_red} no se va a entrenar.")

    # Entrenamiento de la red
    if entrenar_nueva_red:

        ## ERROR INICIAL sobre el test ##
        with torch.no_grad():
            pred_test_init = [NN(x) for x in test_xs]
            loss_init = sum( loss_f(yout, ytrue) for yout,ytrue in zip(pred_test_init, test_ys) ) / len(test_ys) 
        print(f"\nLa red '{nombre_archivo_red}' tiene una perdida inicial sobre el test: {loss_init}")

        ## ENTRENAMIENTO ##
        print(f"\nIniciamos entrenamiento de {stp_n} pasos de la red '{nombre_archivo_red}'.\n")
        NN.train_model(Xs_entrenamiento_def, Ys_entrenamiento_def, stp_n, stp_sz, loss_f, batch_sz)

        ## ERROR FINAL sobre el test ##
        with torch.no_grad():
            pred_test_fin = [NN(x) for x in test_xs]
            loss_final = sum( loss_f(yout, ytrue) for yout,ytrue in zip(pred_test_fin, test_ys) ) / len(test_ys) 
        print(f"\nLa red '{nombre_archivo_red}' tiene una perdida final sobre el test: {loss_final}")

    # Guardado de la red
    if save_new_NN:
        guardar_red(NN, nombre_archivo_red)
    else:
        print(f"Se ha decidido no guardar la red {nombre_archivo_red}.")
