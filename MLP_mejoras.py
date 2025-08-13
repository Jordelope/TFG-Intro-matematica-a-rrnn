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


"""

Fichero para probar posibles mejoras sin modificar el original que funciona correctamente.

PENDIENTE: Mejorar el entrenamiento

IDEA: Es el momento de quitarse las listas y pasar a Vectorización del forward para batch (usando ventajas de pytorch) ?

"""

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

# Diccionario de funciones de activación
activations = {
    "relu": torch.relu,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
    None: lambda x: x  # Sin activación
}


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
    def __init__(self, dim_in : int, f_act : callable=None):
        self.w = torch.randn(dim_in, dtype=torch.float32) / dim_in**0.5
        self.b = torch.randn(1, dtype=torch.float32)
        self.f_act = f_act

    def parameters(self):
        """Devuelve los parámetros entrenables de la neurona (pesos y sesgo)."""
        return [self.w, self.b] # ¿seguro?

    def __call__(self, x):
        """
        Calcula la salida de la neurona para una entrada x.
        torch.dot(self.w, x): producto escalar entre pesos y entrada.
        self.f_act: función de activación aplicada al resultado.
        """
        act = x @ self.w + self.b
        if self.f_act is None:
            return act
        else:
            return self.f_act(act)

## Capa de neuronas ##

class Layer:
    """
    Clase que representa una capa de neuronas.
    - dim_in: número de entradas por neurona (dimension de la entrada).
    - dim_out: número de neuronas en la capa (dimensión de la salida).
    - f_act: función de activación para todas las neuronas de la capa.
    Contiene una lista de objetos Neuron.    ELIMINAR LISTA NEURONAS
    """
    def __init__(self, dim_in : int, dim_out : int, f_act : callable=None):
        self.neurons = [Neuron(dim_in, f_act) for _ in range(dim_out)] # Esto y la clase neurona se vuelven inutiles vectorizando con pytorch
        # Vectorizamos pesos: matriz [nin, nout]
        self.w = torch.randn(dim_in, dim_out, dtype=torch.float32) / dim_in**0.5 # ¿ PREGUNTAR PQ DE ESTA FORMA Y NO  SIMPLEMENTE ENTRE (0,1) ? (MEJOR (0,1) CREO YO)
        self.b = torch.randn(dim_out, dtype=torch.float32) # ¿ me interesa mas sesgos 0 o aleatorios? (si aleatorio mejor (0,1))
        self.f_act = f_act

    def parameters(self):
        """Devuelve todos los parámetros entrenables de la capa."""
        # return [p for neuron in self.neurons for p in neuron.parameters()]   ## Version original ##
        return [self.w, self.b] # <- ¿SEGURO?

    def __call__(self, x):
        """
        Calcula la salida de la capa para una entrada x.
        Devuelve un tensor con la salidas.
        """
        act = x @ self.w +self.b
        if self.f_act is None:
            return act     ## ¿ util poner act[0] if len(act) == 1 else torch.stack(act) ? o no es necesario ? ##
        else: 
            return self.f_act(act)
        ## ¿HAY QUE PONER CASO ESPECIAL PARA F.SOFTMAX U OTRAS FUNCIONES DE ACTIVACION ? CREO QUE SI ##
         

## Red neuronal(perceptron multicapa) compuesta por varias capas ##

class MLP:
    """
    Clase que representa un perceptrón multicapa (MLP).
    - nin: número de entradas.
    - nout: número de salidas.
    - estructura_oct: lista con el número de neuronas en cada capa oculta.

    La red se compone de varias capas (Layer), almacenadas en self.layers.
    El método __call__ permite pasar una entrada por toda la red.
    """
    def __init__(self, 
                 dim_in : int, 
                 dim_out : int, 
                 estructura_oct : list, 
                 f_act_list : list=None)  :
        """
        - nin :  numero de entradas
        - nout : numero de salidas
        - estructura_oct :  lista con el número de neuronas en cada capa oculta.
        - f_act_list: lista de funciones de activacion (debe tener lonitud = len(estructura_oct) + 1)  ¿ +1 o +2 ? (bastante seguro que 1)
        """
        if f_act_list is None:
            f_act_list = [None for i in range( len(estructura_oct)+1 ) ]

        dims = [dim_in] + estructura_oct + [dim_out]
        self.dims = dims
        self.dim_in = dim_in
        self.dim_out  = dim_out
        self.activaciones = f_act_list ## ¿ meter assert para garantizar longitud?
        self.layers = [ Layer(dims[i], dims[i+1], f_act) for i,f_act in zip( range(len(dims) - 1) , self.activaciones) ]
        

    def parameters(self):
        """Devuelve todos los parámetros entrenables de la red (de todas las capas)."""
        return [p for layer in self.layers for p in layer.parameters()]

    def __call__(self, x : torch.Tensor):
        """
        Propaga la entrada x a través de todas las capas de la red.
        """
        ## Asserte de dim? o en enlayer mejor?
        for layer in self.layers:
            x = layer(x)
        return x


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

                ypred_batch = [self(x) for x in X_batch]  ### ¿  SE PUEDE HACER SELF(BATCH) ? ###
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
      - activaciones: nombres de las funciones de activación
      - pesos: lista de todos los parámetros (pesos y sesgos) de la red
    """
    state = [p.detach().tolist() for p in red.parameters()]
    estructura = [len(red.layers[0].neurons[0].w)] + [len(layer.neurons) for layer in red.layers]   ### AAAAAAAAAAQUIIIIII
    estructura = red.dims #mejor asi?
    # Guardamos los nombres de las funciones de activación
    activaciones = [ f_act.__name__ if f_act else None for f_act in red.activaciones] ## ¿ESTA BIEN HECHA ESTA LISTA?

    import json
    with open(archivo, "w") as f:
        json.dump({"estructura": estructura, "activaciones": activaciones, "pesos": state}, f)
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
    activaciones = [nombre_a_func.get(f_act, None) for f_act in data["activaciones"]]  ## REVISAR SI ESTA BIEN ESTA LISTA

    dim_in = estructura[0]
    dim_out = estructura[-1]
    capas = estructura[1:-1]  # Estructura intermedia

    red = MLP(dim_in, dim_out, capas, activaciones)
    flat_params = [p for layer in red.layers for neuron in layer.neurons for p in neuron.parameters()]  ### AAAAAAQUIIIIIII
    flat_params = [p for layer in red.layers for p in layer.parameters()] #revisar que esté bien ?

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
# Las clases están relacionadas jerárquicamente: MLP -> Layer
# Cada nivel encapsula el anterior, permitiendo construir redes de cualquier tamaño y profundidad.
#
# Para entrenar una red, se recomienda preparar los datos como listas de tensores de tipo float32.
#
# Ejemplo de uso:
#   red = MLP(dim_in=10, dim_out=3, estructura=[5, 5])
#   red.train_model(X_train, Y_train, n_steps=100, stp_sz=0.01)
#   guardar_red(red, 'mi_red.json')
#   red2 = cargar_red('mi_red.json')