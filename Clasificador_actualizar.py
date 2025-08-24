from MLP import MLP
from Autoencoder import Autoencoder
from Clasificador import Clasificador
from Guardar_Cargar import guardar_modelo, cargar_modelo

"""
Fichero para actualizar el encoder de un clasificador si se ha mejorado con el entrenamiento del autoencoder original.
"""
## Datos de clasificador y autoecnoder a actualizar ##
clasificador_a_actualizar = r""
autoencoder_mejorado = r""

comentario = ""
añadir_comentario = False  # Opción de añadir comentario a descripción ya existente del clasificador

if __name__=="__main_-":
    clasificador = cargar_modelo(clasificador_a_actualizar)
    autoencoder  = cargar_modelo(autoencoder_mejorado)

    nuevo_encoder = autoencoder.encoder
    clasificador.encoder = nuevo_encoder
    if añadir_comentario:
        clasificador.add_descript(comentario)
    
    print(f"\nSe ha actualizado el clasificador {clasificador_a_actualizar} con el encoder proveniente de {autoencoder_mejorado}.\n")


