"""
MLP.py
-------

Este módulo implementa una red neuronal multicapa (MLP, Multi-Layer Perceptron) desde cero utilizando PyTorch, con un enfoque matemático y didáctico. 
Aquí se definen las clases y funciones necesarias para construir, entrenar, guardar y cargar redes neuronales simples, sin utilizar las abstracciones 
de alto nivel de PyTorch (como nn.Module), para facilitar la comprensión de los fundamentos matemáticos.

Estructura principal del código:
- get_batches: Generador de lotes aleatorios para entrenamiento por lotes.
- Neuron: Implementa una neurona individual con pesos y sesgo.
- Layer: Implementa una capa de neuronas.
- MLP: Implementa una red neuronal multicapa compuesta por varias capas.
- guardar_red / cargar_red: Permiten guardar y cargar redes entrenadas en archivos JSON.

Las clases están relacionadas jerárquicamente: un MLP contiene varias Layer, y cada Layer contiene varias Neuron.
"""

import torch
import torch.nn.functional as F
import random

## Funciones relevantes ##

def get_batches(Xs : list, Ys : list, batch_size : int):
    """
    Genera lotes aleatorios de datos para entrenamiento por lotes (batch training).
    - Xs: lista de entradas.
    - Ys: lista de salidas/targets.
    - batch_size: tamaño del lote.
    Devuelve tuplas (X_batch, Y_batch) de tamaño batch_size.
    """
    n = len(Xs)
    indices = list(range(n))
    random.shuffle(indices)
    for start in range(0, n, batch_size):
        batch_idx = indices[start:start+batch_size]
        yield [Xs[i] for i in batch_idx], [Ys[i] for i in batch_idx]

def crear_tensor(valor):
    """
    Crea un tensor de PyTorch de tipo float32 con gradientes habilitados.
    Esto permite que PyTorch calcule automáticamente los gradientes durante el backpropagation.
    """
    return torch.tensor(valor, dtype=torch.float32, requires_grad=True)

## Neurona individual ##

class Neuron:
    """
    Clase que representa una neurona individual.
    - nin: número de entradas (dimensión de x)
    - f_act: función de activación (por defecto ReLU de torch)
    Cada neurona tiene:
      - w: vector de pesos (tensor)
      - b: sesgo (tensor)
      - f_act: función de activación
    El método __call__ permite usar la neurona como una función: y = f_act(w·x + b)
    """
    def __init__(self, nin : int, f_act : callable=None):
        self.w = crear_tensor([random.uniform(-0.1,0.1) for _ in range(nin)])
        self.b = crear_tensor(random.uniform(-0.1,0.1))
        self.f_act = f_act

    def parameters(self):
        """Devuelve los parámetros entrenables de la neurona (pesos y sesgo)."""
        return [self.w, self.b]

    def __call__(self, x):
        """
        Calcula la salida de la neurona para una entrada x.
        torch.dot(self.w, x): producto escalar entre pesos y entrada.
        self.f_act: función de activación aplicada al resultado.
        """
        act = torch.dot(self.w, x) + self.b
        if self.f_act is None:
            return act
        else:
            return self.f_act(act)

## Capa de neuronas ##

class Layer:
    """
    Clase que representa una capa de neuronas.
    - nin: número de entradas por neurona.
    - nout: número de neuronas en la capa.
    - f_act: función de activación para todas las neuronas de la capa.
    Contiene una lista de objetos Neuron.
    """
    def __init__(self, nin : int, nout : int, f_act : callable=None):
        self.neurons = [Neuron(nin, f_act) for _ in range(nout)]

    def parameters(self):
        """Devuelve todos los parámetros entrenables de la capa (de todas sus neuronas)."""
        return [p for neuron in self.neurons for p in neuron.parameters()]

    def __call__(self, x):
        """
        Calcula la salida de la capa para una entrada x.
        Devuelve un tensor con la salida de cada neurona.
        Si la capa tiene una sola neurona, devuelve un escalar.
        """
        outs = [neuron(x) for neuron in self.neurons]
        return outs[0] if len(outs) == 1 else torch.stack(outs)

## Red neuronal(perceptron multicapa) compuesta por varias capas ##

class MLP:
    """
    Clase que representa un perceptrón multicapa (MLP).
    - nin: número de entradas.
    - nout: número de salidas.
    - estructura: lista con el número de neuronas en cada capa oculta.
    - f_act_salida: función de activación para la capa de salida (por defecto softmax).
    - f_act_oculta: función de activación para las capas ocultas (por defecto ReLU).

    La red se compone de varias capas (Layer), almacenadas en self.layers.
    El método __call__ permite pasar una entrada por toda la red.
    """
    def __init__(self, 
                nin : int, nout : int, estructura : list, 
                f_act_salida : callable=None, f_act_oculta : callable=None)  :
        # Estructura: lista que indica el número de neuronas por capa oculta
        sz = [nin] + estructura + [nout]
        self.layers = [Layer(sz[i], sz[i+1], f_act_oculta) for i in range(len(sz) - 1)]

        self.f_act_salida = f_act_salida
        self.f_act_oculta = f_act_oculta

    def parameters(self):
        """Devuelve todos los parámetros entrenables de la red (de todas las capas)."""
        return [p for layer in self.layers for p in layer.parameters()]

    def __call__(self, x : torch.Tensor):
        """
        Propaga la entrada x a través de todas las capas de la red.
        - Las capas ocultas usan la función de activación oculta.
        - La última capa (salida) es lineal y luego se aplica la función de activación de salida (por defecto softmax).
        """
        for layer in self.layers[:-1]:
            x = layer(x)
        # Última capa lineal (sin activación)
        final_layer = self.layers[-1]
        logits = [torch.dot(neuron.w, x) + neuron.b for neuron in final_layer.neurons]
        logits = torch.stack(logits)
        
        if self.f_act_salida is None:
            return logits
        elif self.f_act_salida == F.softmax:
            return self.f_act_salida(logits, dim=-1)
        else:
            return self.f_act_salida(logits)

    def train_model(self, 
                    training_data : list, target_vector : list, 
                    n_steps : int, stp_sz : float , 
                    loss_f : callable = F.cross_entropy, 
                    batch_size : int=None):
        """
        Entrena la red neuronal usando descenso de gradiente estocástico.
        - training_data: lista de entradas.
        - target_vector: lista de salidas esperadas.
        - n_steps: número de iteraciones de entrenamiento.
        - stp_sz: tamaño del paso (learning rate).
        - loss_f: función de pérdida (por defecto cross_entropy de torch).
        - batch_size: tamaño del lote (si None, usa todo el dataset).

        Para cada lote:
        1. Calcula la predicción de la red para cada entrada del lote.
        2. Calcula la pérdida promedio del lote.
        3. Realiza backpropagation (loss.backward()).
        4. Actualiza los parámetros usando el gradiente calculado.
        """
        if batch_size is None:
            batch_size = len(training_data)

        parameters = self.parameters()
        
        for k in range(n_steps):
            epoch_loss = 0.0
            num_batches = 0
            for X_batch, Y_batch in get_batches(training_data, target_vector, batch_size):
                for p in parameters:
                    if p.grad is not None:
                        p.grad.zero_()

                ypred_batch = [self(x) for x in X_batch]
                loss = sum(loss_f(ypred, ytrue) for ypred, ytrue in zip(ypred_batch, Y_batch)) / len(ypred_batch)
                epoch_loss += loss.item()
                num_batches += 1
                loss.backward()

                for p in parameters:
                    p.data -= stp_sz * p.grad

            if k % 20 == 0 or k == n_steps - 1:
                avg_loss = epoch_loss / num_batches
                print(f"Paso {k} | Pérdida promedio: {avg_loss:.4f}")

# Guardar y cargar un MLP en archivo JSON

def guardar_MLP(red : MLP, archivo : str):
    """
    Guarda el estado de una red MLP en un archivo JSON.
    - red: objeto MLP a guardar.
    - archivo: ruta del archivo destino.
    Se almacena:
      - estructura: lista con el número de neuronas por capa (incluye entrada y salida)
      - f_salida, f_oculta: nombres de las funciones de activación
      - pesos: lista de todos los parámetros (pesos y sesgos) de la red
    """
    state = [p.detach().tolist() for p in red.parameters()]
    estructura = [len(red.layers[0].neurons[0].w)] + [len(layer.neurons) for layer in red.layers]
    # Guardamos los nombres de las funciones de activación
    f_salida = red.f_act_salida.__name__ if red.f_act_salida else None
    f_oculta = red.f_act_oculta.__name__ if red.f_act_oculta else None

    import json
    with open(archivo, "w") as f:
        json.dump({"estructura": estructura, "f_salida": f_salida, "f_oculta": f_oculta, "pesos": state}, f)
    print(f"Red guardada en '{archivo}'\n")

def cargar_MLP(archivo : str):
    """
    Carga una red MLP desde un archivo JSON previamente guardado con guardar_red.
    - archivo: ruta del archivo JSON.
    El archivo debe contener:
      - estructura: lista con el número de neuronas por capa
      - f_salida, f_oculta: nombres de las funciones de activación
      - pesos: lista de parámetros (pesos y sesgos)
    Se reconstruye la red y se restauran los parámetros.
    """
    import json
    with open(archivo, "r") as f:
        data = json.load(f)

    estructura = data["estructura"]
    pesos = data["pesos"]
    nombre_a_func = {
        "relu": F.relu,
        "softmax": F.softmax,
        "tanh": F.tanh,
        "sigmoid": F.sigmoid,
        "log_softmax": F.log_softmax,
        None: None
    }
    f_salida = nombre_a_func.get(data["f_salida"], None)
    f_oculta = nombre_a_func.get(data.get("f_oculta"), None)

    nin = estructura[0]
    nout = estructura[-1]
    capas = estructura[1:-1]  # Estructura intermedia

    red = MLP(nin, nout, capas, f_salida, f_oculta)
    flat_params = [p for layer in red.layers for neuron in layer.neurons for p in neuron.parameters()]

    for param, val in zip(flat_params, pesos):
        val_tensor = torch.tensor(val, dtype=torch.float32)
        if param.shape != val_tensor.shape:
            raise ValueError("La forma del peso cargado no coincide con la esperada")
        param.data = val_tensor

    print(f"Red cargada correctamente desde '{archivo}'")
    return red


# =============================
# Notas finales
# =============================
# Este código busca mostrar cómo funcionan las redes neuronales a bajo nivel usando PyTorch.
# No utiliza nn.Module ni optimizadores de alto nivel para que el proceso matemático sea transparente.
# Las redes se pueden guardar y cargar fácilmente en formato JSON para su reutilización o análisis.
#
# Las funciones de activación y de pérdida utilizadas son las de torch.nn.functional (F), que operan sobre tensores y permiten el cálculo automático de gradientes.
#
# Las clases están relacionadas jerárquicamente: MLP -> Layer -> Neuron.
# Cada nivel encapsula el anterior, permitiendo construir redes de cualquier tamaño y profundidad.
#
# Para entrenar una red, se recomienda preparar los datos como listas de tensores de tipo float32.
#
# Ejemplo de uso:
#   red = MLP(nin=10, nout=3, estructura=[5, 5])
#   red.train_model(X_train, Y_train, n_steps=100, stp_sz=0.01)
#   guardar_red(red, 'mi_red.json')
#   red2 = cargar_red('mi_red.json')