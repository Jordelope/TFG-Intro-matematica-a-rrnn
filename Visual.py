import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from MLP import MLP, guardar_MLP, cargar_MLP
from Autoencoder import Autoencoder, guardar_autoencoder, cargar_autoencoder
#--------------------------------------------------------------------------
from proc_datos_modular import Xs_test_def, nombre_set_test

"""
Objetivo del fichero: 
Función visual que tome un autoencoder y un conjunto de vectores y muestre una representacion visual de la codificación de estos en el espacio latente.
"""

def visual(autoencoder_nombre : str, 
           datos : list, 
           dim: int,
           etiquetas : list = None,
           titulo_graf : str = None
           ):
    
    print(f"Se van a visualizar {len(datos)} vectores en el espacio latente de dimension {dim} dado por el autoencoder {autoencoder_nombre}.")

    autoencoder = cargar_autoencoder(autoencoder_nombre)
    with torch.no_grad():
        data_codificado = [autoencoder.encoder(x).detach().numpy() for x in datos ]
    
    if len(data_codificado[0]) > dim:
        
        print("La función no esta aun preparada para llevar a cabo un reduccion de la dimensionalidad. \nPor favor introduzca una dimension que coincide con la del espacio latente de su autoencoder")

    elif dim == 2: # añadir etiquetas y titulo grafico
        fig, ax = plt.subplots()
        
        data_x = [x[0] for x in data_codificado]
        data_y = [x[1] for x in data_codificado]
        
        ax.scatter(data_x , data_y)
        plt.show()

    elif dim == 3: # añadir etiquetas y titulo grafico
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        data_x = [x[0] for x in data_codificado]
        data_y = [x[1] for x in data_codificado]
        data_z = [x[2] for x in data_codificado]
        
        ax.scatter(data_x , data_y, data_z)
        plt.show()

    else:
        print("La función no esta aun preparada para representar en dimensiones mayores que 3")




#------------------------------------------------------------------------------------------------------------

## PRUEBAS ##

autoencoder_dim2 = r"redes_disponibles\pruebaVisual_dim2_autoencod.json"
autoencoder_dim3 = r"redes_disponibles\pruebaVisual_dim3_autoencod.json"
vectores = Xs_test_def

visual(autoencoder_dim2,vectores,2)
visual(autoencoder_dim3,vectores,3)

