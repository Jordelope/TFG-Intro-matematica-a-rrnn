import torch
from Clasificador import Clasificador, guardar_classificador, cargar_classificador
from Autoencoder import Autoencoder, guardar_autoencoder, cargar_autoencoder
from MLP import MLP
import os

def test_actualizar_encoder():
    # Crear encoder y decoder originales
    encoder_orig = MLP(10, 4, [8], [torch.relu, torch.relu])
    decoder_orig = MLP(4, 10, [8], [torch.relu, torch.relu])
    autoencoder_orig = Autoencoder(encoder_orig, decoder_orig)
    guardar_autoencoder(autoencoder_orig, 'autoencoder_test_orig.json')

    # Crear encoder mejorado (diferente estructura)
    encoder_mejorado = MLP(10, 4, [12], [torch.relu, torch.relu])
    decoder_mejorado = MLP(4, 10, [12], [torch.relu, torch.relu])
    autoencoder_mejorado = Autoencoder(encoder_mejorado, decoder_mejorado)
    guardar_autoencoder(autoencoder_mejorado, 'autoencoder_test_mejorado.json')

    # Crear clasificador con encoder original
    clasificador = Clasificador(encoder_orig, n_classes=3, estr_oc_clas=[6], list_act_clas=[torch.relu, None])
    guardar_classificador(clasificador, 'clasificador_test.json')

    # Cargar modelos desde disco
    clasificador_cargado = cargar_classificador('clasificador_test.json')
    autoencoder_mejorado_cargado = cargar_autoencoder('autoencoder_test_mejorado.json')

    # Actualizar encoder del clasificador
    clasificador_cargado.encoder = autoencoder_mejorado_cargado.encoder

    # Comprobar que la estructura del encoder ha cambiado
    assert clasificador_cargado.encoder.dims == encoder_mejorado.dims, "El encoder no se ha actualizado correctamente."
    print("Test pasado: El encoder del clasificador se ha actualizado correctamente.")

    # Limpiar archivos temporales
    for f in ['autoencoder_test_orig.json', 'autoencoder_test_mejorado.json', 'clasificador_test.json']:
        if os.path.exists(f):
            os.remove(f)

if __name__ == "__main__":
    test_actualizar_encoder()
