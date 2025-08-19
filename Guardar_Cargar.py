import torch
from MLP import MLP, cargar_MLP, guardar_MLP
from Autoencoder import Autoencoder, guardar_autoencoder, cargar_autoencoder
from Clasificador import Clasificador, guardar_classificador, cargar_classificador


def guardar_modelo(modelo,nombre_archivo):
    
    # Revisamos que no se vayan a guardar parametro con valor Nan
    parametros = modelo.parameters()
    if any(torch.isnan(p).any() for p in parametros):
        raise ValueError(" Se han detectado valores Nan en el modelo y por tanto no se va a guardar.")
    
    # Guardamos segun tipo de modelo
    if isinstance(modelo, MLP):
        guardar_MLP(modelo, nombre_archivo)
    elif isinstance(modelo, Autoencoder):
        guardar_autoencoder(modelo, nombre_archivo)
    elif isinstance(modelo, Clasificador):
        guardar_classificador(modelo, nombre_archivo)
    else:
        raise TypeError("No se reconoce ningún tipo de red.")

def cargar_modelo(archivo):
    import json
    with open(archivo, "r") as f:
        data = json.load(f)
        tipo_modelo = data["tipo_modelo"]

    if tipo_modelo == "mlp":
        modelo = cargar_MLP(archivo)

    elif tipo_modelo == "autoencoder":
        modelo = cargar_autoencoder(archivo)

    elif tipo_modelo == "clasificador":
        modelo = cargar_classificador(archivo)
    
    else:
        raise ValueError("No se reconoce el tipo de modelo.")
    
    return modelo