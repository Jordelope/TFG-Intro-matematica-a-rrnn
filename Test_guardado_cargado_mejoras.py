import torch
import torch.nn.functional as F
import os
from MLP_mejoras import MLP, cargar_MLP, guardar_MLP
from Autoencoder_mejoras import Autoencoder, cargar_autoencoder, guardar_autoencoder
from Clasificador import Clasificador, guardar_classificador, cargar_classificador

import tempfile
import shutil
import numpy as np

def tensors_equal(t1, t2):
    return torch.allclose(t1, t2, atol=1e-6)

def mlp_equal(mlp1, mlp2):
    # Estructura
    if len(mlp1.layers) != len(mlp2.layers):
        return False
    for l1, l2 in zip(mlp1.layers, mlp2.layers):
        if len(l1.neurons) != len(l2.neurons):
            return False
        for n1, n2 in zip(l1.neurons, l2.neurons):
            if not tensors_equal(n1.w, n2.w) or not tensors_equal(n1.b, n2.b):
                return False
    # Funciones de activación
    if (mlp1.f_act_salida is None) != (mlp2.f_act_salida is None):
        return False
    if (mlp1.f_act_oculta is None) != (mlp2.f_act_oculta is None):
        return False
    if mlp1.f_act_salida and mlp2.f_act_salida:
        if mlp1.f_act_salida.__name__ != mlp2.f_act_salida.__name__:
            return False
    if mlp1.f_act_oculta and mlp2.f_act_oculta:
        if mlp1.f_act_oculta.__name__ != mlp2.f_act_oculta.__name__:
            return False
    return True

def autoencoder_equal(ae1, ae2):
    return mlp_equal(ae1.encoder, ae2.encoder) and mlp_equal(ae1.decoder, ae2.decoder)


def clasificador_equal(c1, c2):
    # n_classes y dim_latente
    if c1.n_classes != c2.n_classes:
        print(f"\nDiferencia en n_classes: {c1.n_classes} vs {c2.n_classes}\n")
        return False
    if c1.dim_latente != c2.dim_latente:
        print(f"\nDiferencia en dim_latente: {c1.dim_latente} vs {c2.dim_latente}\n")
        return False
    # encoder
    if not mlp_equal(c1.encoder, c2.encoder):
        print("\nDiferencia en encoder:\n")
        # Comparación detallada de layers y pesos
        for i, (l1, l2) in enumerate(zip(c1.encoder.layers, c2.encoder.layers)):
            if len(l1.neurons) != len(l2.neurons):
                print(f"\n  Capa {i}: diferente número de neuronas: {len(l1.neurons)} vs {len(l2.neurons)}\n")
            for j, (n1, n2) in enumerate(zip(l1.neurons, l2.neurons)):
                if not tensors_equal(n1.w, n2.w):
                    print(f"\n  Encoder capa {i} neurona {j}: pesos diferentes\n")
                if not tensors_equal(n1.b, n2.b):
                    print(f"\n  Encoder capa {i} neurona {j}: bias diferente\n")
        # Funciones de activación
        if (c1.encoder.f_act_salida is not None and c2.encoder.f_act_salida is not None and c1.encoder.f_act_salida.__name__ != c2.encoder.f_act_salida.__name__):
            print(f"\n  Encoder función de activación de salida: {c1.encoder.f_act_salida.__name__} vs {c2.encoder.f_act_salida.__name__}\n")
        if (c1.encoder.f_act_oculta is not None and c2.encoder.f_act_oculta is not None and c1.encoder.f_act_oculta.__name__ != c2.encoder.f_act_oculta.__name__):
            print(f"\n  Encoder función de activación oculta: {c1.encoder.f_act_oculta.__name__} vs {c2.encoder.f_act_oculta.__name__}\n")
        return False
    # mlp_clasificador
    if not mlp_equal(c1.mlp_clasificador, c2.mlp_clasificador):
        print("\nDiferencia en mlp_clasificador:\n")
        for i, (l1, l2) in enumerate(zip(c1.mlp_clasificador.layers, c2.mlp_clasificador.layers)):
            if len(l1.neurons) != len(l2.neurons):
                print(f"\n  Capa {i}: diferente número de neuronas: {len(l1.neurons)} vs {len(l2.neurons)}\n")
            for j, (n1, n2) in enumerate(zip(l1.neurons, l2.neurons)):
                if not tensors_equal(n1.w, n2.w):
                    print(f"\n  Clasificador capa {i} neurona {j}: pesos diferentes\n")
                if not tensors_equal(n1.b, n2.b):
                    print(f"\n  Clasificador capa {i} neurona {j}: bias diferente\n")
        # Funciones de activación
        if (c1.mlp_clasificador.f_act_salida is not None and c2.mlp_clasificador.f_act_salida is not None and c1.mlp_clasificador.f_act_salida.__name__ != c2.mlp_clasificador.f_act_salida.__name__):
            print(f"\n  Clasificador función de activación de salida: {c1.mlp_clasificador.f_act_salida.__name__} vs {c2.mlp_clasificador.f_act_salida.__name__}\n")
        if (c1.mlp_clasificador.f_act_oculta is not None and c2.mlp_clasificador.f_act_oculta is not None and c1.mlp_clasificador.f_act_oculta.__name__ != c2.mlp_clasificador.f_act_oculta.__name__):
            print(f"\n  Clasificador función de activación oculta: {c1.mlp_clasificador.f_act_oculta.__name__} vs {c2.mlp_clasificador.f_act_oculta.__name__}\n")
        return False
    return True

def test_guardado_cargado_MLP():
    mlp = MLP(5, 2, [4, 3], f_act_salida=F.softmax, f_act_oculta=F.relu)
    # Modificamos pesos para testear
    for p in mlp.parameters():
        p.data += torch.randn_like(p) * 0.01
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'mlp.json')
        guardar_MLP(mlp, path)
        mlp2 = cargar_MLP(path)
        assert mlp_equal(mlp, mlp2), 'MLP no es igual tras guardar y cargar'
    print('Test MLP guardado/cargado: OK')

def test_guardado_cargado_Autoencoder():
    encoder = MLP(6, 3, [5], f_act_salida=F.tanh, f_act_oculta=F.relu)
    decoder = MLP(3, 6, [4], f_act_salida=None, f_act_oculta=F.sigmoid)
    ae = Autoencoder(encoder, decoder)
    # Modificamos pesos
    for p in ae.parameters():
        p.data += torch.randn_like(p) * 0.01
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'ae.json')
        guardar_autoencoder(ae, path)
        ae2 = cargar_autoencoder(path)
        assert autoencoder_equal(ae, ae2), 'Autoencoder no es igual tras guardar y cargar'
    print('Test Autoencoder guardado/cargado: OK')

def test_guardado_cargado_Clasificador():
    encoder = MLP(8, 4, [7], f_act_salida=F.relu, f_act_oculta=F.tanh)
    mlp_cls = MLP(4, 3, [5], f_act_salida=F.softmax, f_act_oculta=F.relu)
    # Modificamos pesos
    for p in encoder.parameters():
        p.data += torch.randn_like(p) * 0.01
    for p in mlp_cls.parameters():
        p.data += torch.randn_like(p) * 0.01
    clasif = Clasificador(encoder, 3, [5], f_act_salida=F.softmax, f_act_oculta=F.relu)
    clasif.mlp_clasificador = mlp_cls
    clasif.layers = encoder.layers + mlp_cls.layers
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'clasif.json')
        guardar_classificador(clasif, path)
        clasif2 = cargar_classificador(path)
        assert clasificador_equal(clasif, clasif2), 'Clasificador no es igual tras guardar y cargar'
    print('Test Clasificador guardado/cargado: OK')

def test_consistencia_componentes():
    # Un mismo MLP como encoder de autoencoder y clasificador, y como MLP suelto
    encoder = MLP(10, 5, [8], f_act_salida=F.tanh, f_act_oculta=F.relu)
    decoder = MLP(5, 10, [7], f_act_salida=None, f_act_oculta=F.sigmoid)
    mlp_cls = MLP(5, 4, [6], f_act_salida=F.softmax, f_act_oculta=F.relu)
    # Modificamos pesos
    for p in encoder.parameters():
        p.data += torch.randn_like(p) * 0.01
    for p in decoder.parameters():
        p.data += torch.randn_like(p) * 0.01
    for p in mlp_cls.parameters():
        p.data += torch.randn_like(p) * 0.01
    ae = Autoencoder(encoder, decoder)
    clasif = Clasificador(encoder, 4, [6], f_act_salida=F.softmax, f_act_oculta=F.relu)
    clasif.mlp_clasificador = mlp_cls
    clasif.layers = encoder.layers + mlp_cls.layers
    with tempfile.TemporaryDirectory() as tmpdir:
        # Guardar encoder como MLP suelto
        path_encoder = os.path.join(tmpdir, 'encoder.json')
        guardar_MLP(encoder, path_encoder)
        encoder2 = cargar_MLP(path_encoder)
        # Guardar autoencoder
        path_ae = os.path.join(tmpdir, 'ae.json')
        guardar_autoencoder(ae, path_ae)
        ae2 = cargar_autoencoder(path_ae)
        # Guardar clasificador
        path_cls = os.path.join(tmpdir, 'cls.json')
        guardar_classificador(clasif, path_cls)
        clasif2 = cargar_classificador(path_cls)
        # El encoder cargado como MLP, como encoder de AE y de Clasificador deben ser iguales
        assert mlp_equal(encoder, encoder2), 'Encoder MLP no igual tras guardar/cargar'
        assert mlp_equal(encoder, ae2.encoder), 'Encoder de Autoencoder no igual al original'
        assert mlp_equal(encoder, clasif2.encoder), 'Encoder de Clasificador no igual al original'
        # El decoder del AE debe ser igual tras guardar/cargar
        assert mlp_equal(decoder, ae2.decoder), 'Decoder de Autoencoder no igual tras guardar/cargar'
        # El mlp_clasificador debe ser igual tras guardar/cargar
        assert mlp_equal(mlp_cls, clasif2.mlp_clasificador), 'MLP clasificador no igual tras guardar/cargar'
    print('Test consistencia componentes compartidos: OK')

if __name__ == '__main__':
    test_guardado_cargado_MLP()
    test_guardado_cargado_Autoencoder()
    test_guardado_cargado_Clasificador()
    test_consistencia_componentes()
