# Redes Neuronales desde Cero con PyTorch (Enfoque Matemático)

Este proyecto es una introducción práctica y matemática a las redes neuronales, implementando desde cero modelos como perceptrones multicapa (MLP) y autoencoders utilizando únicamente tensores y funciones de PyTorch. El objetivo es facilitar la comprensión de los fundamentos matemáticos y computacionales de las redes neuronales, evitando el uso de abstracciones de alto nivel como `nn.Module`.

## Objetivo

- Enseñar cómo funcionan internamente las redes neuronales, sus cálculos y entrenamiento, usando PyTorch solo como motor de tensores y funciones matemáticas.
- Proveer ejemplos claros y didácticos para estudiantes o personas que quieran entender los fundamentos antes de usar frameworks de alto nivel.

## Estructura y resumen de archivos

- **MLP.py**: Implementa desde cero una red neuronal multicapa (MLP), con clases para neuronas, capas y la red completa. Incluye funciones para entrenar, guardar y cargar modelos en formato JSON.
- **Autoencoder.py**: Define la clase `Autoencoder` usando dos MLP (encoder y decoder). Permite entrenar, guardar y cargar autoencoders personalizados.
- **proc_datos_modular.py**: Procesa y normaliza los datos de entrada y salida para entrenamiento y test, permitiendo seleccionar columnas, normalización y filtrado. Devuelve tensores listos para usar en los modelos.
- **crear_autoencoder.py**: Script para crear, entrenar y guardar un autoencoder. Permite definir la arquitectura y los hiperparámetros, y utiliza los datos procesados.
- **crear_MLP.py**: Permite crear y guardar una red MLP personalizada, definiendo arquitectura y funciones de activación. Incluye utilidades para evaluar la red sobre datos de test.
- **test_procesado_datos.py**: Incluye tests automáticos para verificar que el procesamiento de datos funciona correctamente tanto para autoencoders como para clasificación.
- **entrenar_MLP.py**: Script para cargar, entrenar y evaluar una red MLP sobre un conjunto de datos. Permite guardar la red solo si mejora el error.
- **usar_red_nba.py**, **procesar_datos_entrenamiento.py**, **entrenar_MLP_nba.py**: Versiones iniciales o experimentales. Pueden dar errores ya que no han sido actualizados tras los últimos cambios en el resto del código.

## Requisitos

Instala las dependencias ejecutando:

```
pip install -r requirements.txt
```

## Notas
- Los datasets y redes guardadas se encuentran en las carpetas `datasets/` y `redes_disponibles/` respectivamente.
- El código está pensado para ser didáctico y transparente, ideal para quienes quieren aprender los fundamentos matemáticos y computacionales de las redes neuronales.
