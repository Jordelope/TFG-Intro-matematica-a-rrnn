import torch
import torch.nn.functional as F
import numpy
from MLP import MLP, cargar_MLP, guardar_MLP, nombre_a_func
from Procesamiento_datos_modular import Xs_entrenamiento_def, Xs_test_def, Ys_entrenamiento_def, Ys_test_def
from Autoencoder import Autoencoder, guardar_autoencoder, cargar_autoencoder
from Clasificador import Clasificador, guardar_classificador, cargar_classificador
from Visual import visual


"""
Fichero para usar un clasificador sobre datos. La idea es que pueda:
    - Clasificar unos datos sin mas
    - Clasificar midiendo rendimiento frente a un test
    - Mostrar su representacion de los datos en espacio latente
"""

"""
Mejoras a hacer:
    - opcion de elegir el dataset desde este fichero
    - mejoras procesado de datoss
    - dejar datos procesados guardados para no hacerlo cada vez
"""

## Funciones_relevantes ##
def vector_a_clase(xs):
    return xs.argmax().item()

## NOMBRE Clasificador ##
archivo_clasificador = r"redes_disponibles\intento1_Clasificador.json"
ver_solo_decision = True # Si es True veremos SOLO que clase asigna a los datos el clasificador. En caso contrario veremos las "Probabilidades".

## DATOS a clasificar / TEST a evaluar ##
modo_evaluacion = True

xs_a_clasificar = Xs_test_def
ys_para_evaluar = Ys_test_def


## OPCIONES visualizacion ##
ver_repr_latente = True

dim_repr = 3 
modos_redd_dim = ["pca", "tsne", "umap"]
modo_redd_dim = "umap"
titulo_grafico = "Titulo"


#------------------------------------------------------------------------------------------------------------------------------------------------------


if __name__=="__main__":
    
    # Cargamos Clasificador
    print(f"Se va a usar el Clasificador: {archivo_clasificador}.\n")
    clasificador = cargar_classificador(archivo_clasificador)

    if modo_evaluacion :
        print("\nEstamos en modo Evaluacion.\n")
        clases_test = [vector_a_clase(x) for x in ys_para_evaluar]
    
    # Pasamos datos por el clasificador y extraemos su clase
    datos_clasificados = clasificador(xs_a_clasificar).detach().numpy()
    clases_datos = [vector_a_clase(xs) for xs in datos_clasificados]
    # Mostramos resultados del clasificador
    print(f"El clasificador asigna a los datos las siguientes clases:\n{clases_datos}\n")
    if not ver_solo_decision:
        print(f"Con unas probabilidades asignadas:\n {datos_clasificados}\n")
    

    if modo_evaluacion:
        # Comparamos resultado con el test
        cont_correctos = 0
        for i in range(len(clases_test)):
            if clases_test[i] == clases_datos[i]:
                cont_correctos += 1
        accuracy = cont_correctos / len(clases_test)

        print(f"Segun el test, las clases correctas son: \n{clases_test}\n")
        print(f"El modelo {archivo_clasificador} ha acertado {cont_correctos} / {len(clases_test)}.")
        print(f"Precisión en test: {accuracy*100:.2f}%\n")

    
    if ver_repr_latente:
        visual(archivo_clasificador,xs_a_clasificar,dim_repr,clases_datos,titulo_grafico,modo_redd_dim)
        visual(archivo_clasificador,xs_a_clasificar,dim_repr,clases_test,"Clases test",modo_redd_dim)


    


