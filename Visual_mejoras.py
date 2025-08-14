import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from MLP_mejoras import MLP, guardar_MLP, cargar_MLP
from Autoencoder_mejoras import Autoencoder, guardar_autoencoder, cargar_autoencoder
#--------------------------------------------------------------------------
from Procesamiento_datos_modular import Xs_test_def, etiquetas_test, nombre_set_test

"""
Objetivo del fichero: 
Función visual que tome un autoencoder y un conjunto de vectores y muestre una representacion visual de la codificación de estos en el espacio latente.
"""

def reducir_dimensionalidad(data, dim_objetivo=2, metodo="pca"):
    metodo = metodo.lower()
    if metodo == "pca":
        return PCA(n_components=dim_objetivo).fit_transform(data)
    elif metodo == "tsne":
        return TSNE(n_components=dim_objetivo, init='random', random_state=42).fit_transform(data)
    elif metodo == "umap":
        reducer = umap.UMAP(n_components=dim_objetivo, random_state=42)
        return reducer.fit_transform(data)
    else:
        raise ValueError(f"Método de reducción no reconocido: {metodo}. Opciones: pca, tsne, umap")


def visual(autoencoder_nombre : str, 
           datos : list, 
           dim: int = None,
           etiquetas : list = None,
           titulo_graf : str = None,
           metodo_reduccion: str = "pca"
           ):
    
    autoencoder = cargar_autoencoder(autoencoder_nombre)
    with torch.no_grad():
        data_codificado = autoencoder.encoder(datos).detach().numpy()  #conseguimos representacion vectores en esp latente
        dim_latente = autoencoder.dim_latente


    if dim is None: #ajustamos dimension si es necesario
        dim = dim_latente


    if etiquetas is not None: #preparamos etiquetas numericas y con nombre(si tienen)
        clases_num, clases_name = pd.factorize(pd.Series(etiquetas))


    print(f"Se van a visualizar {datos.shape[0]} vectores en el espacio latente de dimension {dim} dado por el autoencoder {autoencoder_nombre}.\n")


    if dim_latente > dim:
        print(f"Reduciendo dimensionalidad de {dim_latente} → {dim} usando {metodo_reduccion.upper()}...\n")
        data_codificado = reducir_dimensionalidad(data_codificado, dim_objetivo=dim, metodo=metodo_reduccion)


    if dim == 2: # añadir etiquetas y titulo grafico
        fig, ax = plt.subplots()
        
        data_x = [x[0] for x in data_codificado]
        data_y = [x[1] for x in data_codificado]
        
        if etiquetas:
            scatter =ax.scatter(data_x, data_y, c = clases_num, cmap= "tab10", alpha =0.8)
            if clases_name is not None:
                handles = []
                for i, clase in enumerate(clases_name):
                    color = scatter.cmap(scatter.norm(i))
                    handles.append(plt.Line2D([0], [0],
                                            marker='o',
                                            color='w',
                                            markerfacecolor=color,
                                            markersize=8,
                                            label=clase))
                ax.legend(handles=handles, title="Clases", loc="best")
            
        else:
            ax.scatter(data_x, data_y)
        
        ax.set_xlabel("Latente 1")
        ax.set_ylabel("Latente 2")
        ax.set_title(titulo_graf)
        plt.show()


    elif dim == 3: # añadir etiquetas y titulo grafico
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        data_x = [x[0] for x in data_codificado]
        data_y = [x[1] for x in data_codificado]
        data_z = [x[2] for x in data_codificado]
        
        if etiquetas:
            scatter = ax.scatter(data_x, data_y, data_z, c = clases_num, cmap= "tab10", alpha =0.8)
            if clases_name is not None:
                handles = []
                for i, clase in enumerate(clases_name):
                    color = scatter.cmap(scatter.norm(i))
                    handles.append(plt.Line2D([0], [0],
                                            marker='o',
                                            color='w',
                                            markerfacecolor=color,
                                            markersize=8,
                                            label=clase))

                ax.legend(handles=handles, title="Clases", loc="best")
        
        else:
            ax.scatter(data_x , data_y, data_z)
        
        ax.set_xlabel("Latente 1")
        ax.set_ylabel("Latente 2")
        ax.set_zlabel("Latente 3")
        ax.set_title(titulo_graf)
        plt.show()
    
    else:
        print("La función no puede representar en dimensiones mayores que 3")




#------------------------------------------------------------------------------------------------------------

## PRUEBAS ##

autoencoder_dim2 =  r"redes_disponibles\mejoras\visual_pruebas_dim2_autoenc.json"
autoencoder_dim3 =  r"redes_disponibles\mejoras\visual_pruebas_dim3_autoenc.json"
autoencoder_dim6 =  r"redes_disponibles\mejoras\visual_pruebas_dim6_autoenc.json"

vectores = Xs_test_def
etiquetas = etiquetas_test
titulo ="PRUEBAS"

if __name__ == "__main__":
    
    visual(autoencoder_dim2,vectores,2,etiquetas,titulo)
    visual(autoencoder_dim3,vectores,3,etiquetas,titulo)
    visual(autoencoder_dim6,vectores,3,etiquetas,titulo,"pca")
    visual(autoencoder_dim6,vectores,3,etiquetas,titulo,"tsne")
    visual(autoencoder_dim6,vectores,3,etiquetas,titulo,"umap") # salta warning


