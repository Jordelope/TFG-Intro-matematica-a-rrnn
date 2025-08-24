import torch
import torch.nn.functional as F
import numpy
from MLP import MLP, nombre_a_func
from Autoencoder import Autoencoder 
from Clasificador import Clasificador 
from Guardar_Cargar import guardar_modelo, cargar_modelo
from Procesar_datos import procesar_datos
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
archivo_clasificador = r"redes_disponibles/clasificador_nba2_mejoras.json"
ver_descripcion = True
ver_solo_decision = True # Si es True veremos SOLO que clase asigna a los datos el clasificador. En caso contrario veremos las "Probabilidades".


## DATOS a clasificar / TEST a evaluar ##
modo_evaluacion = True

xs_a_clasificar = None
ys_para_evaluar = None

archivo_entrenamiento = r"datasets\nba\combined19_25_pergame_filtered.csv"
archivo_test = r"datasets\nba\nba24_25_pergame.csv"
_, _, _, xs_a_clasificar, ys_para_evaluar, etiquetas_test = procesar_datos(archivo_set_train=archivo_entrenamiento,
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

## OPCIONES visualizacion ##
ver_repr_latente = True

dim_repr = 3 
modos_redd_dim = ["pca", "tsne", "umap"]
modo_redd_dim = "pca"
visualizar_todos = True
titulo_grafico = "Titulo"


#------------------------------------------------------------------------------------------------------------------------------------------------------


if __name__=="__main__":
    
    # Cargamos Clasificador
    print(f"Se va a usar el Clasificador: {archivo_clasificador}.\n")
    clasificador = cargar_modelo(archivo_clasificador)
    if ver_descripcion:
        print(f"La descripcion de este modelo nos dice: \n'{clasificador.description}'")

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
    
    #  Comparacion con el test
    if modo_evaluacion:
        cont_correctos = 0
        for i in range(len(clases_test)):
            if clases_test[i] == clases_datos[i]:
                cont_correctos += 1
        accuracy = cont_correctos / len(clases_test)

        print(f"Segun el test, las clases correctas son: \n{clases_test}\n")
        print(f"El modelo {archivo_clasificador} ha acertado {cont_correctos} / {len(clases_test)}.")
        print(f"Precisión en test: {accuracy*100:.2f}%\n")

    # Visualización
    if ver_repr_latente:
        if visualizar_todos:
            visual(archivo_clasificador,xs_a_clasificar,dim_repr,clases_datos,titulo_grafico,"pca")
            if modo_evaluacion:
                visual(archivo_clasificador,xs_a_clasificar,dim_repr,etiquetas_test,"Clases test pca","pca")
            
            visual(archivo_clasificador,xs_a_clasificar,dim_repr,clases_datos,titulo_grafico,"tsne")
            if modo_evaluacion:
                visual(archivo_clasificador,xs_a_clasificar,dim_repr,etiquetas_test,"Clases test tsne","tsne")
            
            visual(archivo_clasificador,xs_a_clasificar,dim_repr,clases_datos,titulo_grafico,"umap")
            if modo_evaluacion:
                visual(archivo_clasificador,xs_a_clasificar,dim_repr,etiquetas_test,"Clases test umap","umap")
        

        else:
            visual(archivo_clasificador,xs_a_clasificar,dim_repr,clases_datos,titulo_grafico,modo_redd_dim)
            if modo_evaluacion:
                visual(archivo_clasificador,xs_a_clasificar,dim_repr,etiquetas_test,f"Clases test {modo_redd_dim}",modo_redd_dim)


    


