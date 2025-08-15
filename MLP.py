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

PENDIENTE: 
    -Mejorar el entrenamiento
    -REVISAR cross_entropy

OPCIONAL: Modificar get_batches para que devuelva tensores (ya lo hace el entrenamiento ahora mismo)

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

nombre_a_func = {
    "relu": torch.relu,
    "softmax": F.softmax,
    "log_softmax": F.log_softmax,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
    "cross_entropy": F.cross_entropy,
    "binary_cross_entropy": F.binary_cross_entropy,
    "mse_loss": F.mse_loss,
    "none": None
    }

## Capa de neuronas ##

class Layer:
    """
    Clase que representa una capa de neuronas.
    - dim_in: número de entradas por neurona (dimension de la entrada).
    - dim_out: número de neuronas en la capa (dimensión de la salida).
    - f_act: función de activación para todas las neuronas de la capa.
    """
    def __init__(self, dim_in : int, dim_out : int, f_act : callable=None):
        # Vectorizamos pesos: matriz [dim_in, dim_out]
        self.w = torch.empty(dim_in, dim_out, dtype=torch.float32, requires_grad=True)  
        self.b = torch.zeros(dim_out, dtype=torch.float32, requires_grad=True) 

        if f_act in (torch.tanh, torch.sigmoid):
            torch.nn.init.xavier_uniform_(self.w)
        elif f_act in (torch.relu, F.relu, F.leaky_relu):
            torch.nn.init.kaiming_uniform_(self.w, nonlinearity="relu")
        else:
            # Por defecto Xavier
            torch.nn.init.xavier_uniform_(self.w)
       
        self.f_act = f_act
        

    def parameters(self):
        """Devuelve todos los parámetros entrenables de la capa."""
        return [self.w, self.b]

    def __call__(self, x):
        """
        Calcula la salida de la capa para una entrada x.
        Devuelve un tensor con la salidas.
        """
        act = x @ self.w + self.b

        if self.f_act is None:
            return act
        elif self.f_act is F.softmax or self.f_act is F.log_softmax:
            return self.f_act(act, dim=-1) 
        else: 
            return self.f_act(act)
         

## Red neuronal(perceptron multicapa) compuesta por varias capas ##

class MLP:
    """
    Clase que representa un perceptrón multicapa (MLP).
    - dim_in: dimension de las entradas.
    - dim_out: dimension de las salidas.
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
        - dim_in :  numero de entradas
        - dim_out : numero de salidas
        - estructura_oct :  lista con el número de neuronas en cada capa oculta.
        - f_act_list: lista de funciones de activacion (debe tener lonitud = len(estructura_oct) + 1)
        """
        if f_act_list is None:
            f_act_list = [None for i in range( len(estructura_oct)+1 ) ]

        dims = [dim_in] + estructura_oct + [dim_out]
        assert len(f_act_list) == len(dims) -1

        self.dims = dims
        self.dim_in = dim_in
        self.dim_out  = dim_out
        self.activaciones = f_act_list
        self.layers = [ Layer(dims[i], dims[i+1], f_act) for i,f_act in zip( range(len(dims) - 1) , self.activaciones) ] 
        

    def parameters(self):
        """Devuelve todos los parámetros entrenables de la red (de todas las capas)."""
        return [p for layer in self.layers for p in layer.parameters()]

    def __call__(self, x : torch.Tensor):
        """
        Propaga la entrada x a través de todas las capas de la red.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0) 
            
        for layer in self.layers:
            x = layer(x)
        return x

    def train_model(self, 
                    training_data : list[torch.Tensor], target_vector : list[torch.Tensor], 
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
               
                # Tensores
                X_batch = torch.stack(X_batch)  # (B, dim_in)
                if loss_f is F.cross_entropy:
                    # Y_batch: índices de clase
                    #Y_batch = torch.tensor(Y_batch, dtype=torch.long)
                    Y_batch = torch.stack(Y_batch)
                    pass
                else:
                    Y_batch = torch.stack(Y_batch)
                
                # Gradientes a 0
                for p in parameters:
                    if p.grad is not None:
                        p.grad.zero_()
                
                # Forward
                ypred_batch = self(X_batch)  
                loss = loss_f(ypred_batch,Y_batch) 
                epoch_loss += loss.item()
                num_batches += 1
                
                # Backward 
                loss.backward()
                
                with torch.no_grad(): # Evitar cosas raras
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
    estructura = red.dims 
    # Guardamos los nombres de las funciones de activación
    activaciones = [ getattr(f_act, "__name__", "none") if f_act else "none" for f_act in red.activaciones] 

    import json
    with open(archivo, "w") as f:
        json.dump({"tipo_modelo":"mlp", "estructura": estructura, "activaciones": activaciones, "pesos": state}, f)
    print(f"Modelo guardado en '{archivo}'\n")


def cargar_MLP(archivo : str):
    """
    Carga una red MLP desde un archivo JSON previamente guardado con guardar_red.
    - archivo: ruta del archivo JSON.
    El archivo debe contener:
      - estructura: lista con el número de neuronas por capa
      - activaciones: lista de las funciones de activación
      - pesos: lista de parámetros (pesos y sesgos)
    Se reconstruye la red y se restauran los parámetros.
    """
    import json
    with open(archivo, "r") as f:
        data = json.load(f)

    estructura = data["estructura"]
    pesos = data["pesos"]
    activaciones = [nombre_a_func.get(f_act, None) for f_act in data["activaciones"]] 
    dim_in = estructura[0]
    dim_out = estructura[-1]
    capas = estructura[1:-1]  # Estructura intermedia

    red = MLP(dim_in, dim_out, capas, activaciones)
    flat_params = [p for layer in red.layers for p in layer.parameters()]

    for param, val in zip(flat_params, pesos):
        val_tensor = torch.tensor(val, dtype=torch.float32)
        if param.shape != val_tensor.shape:
            raise ValueError("La forma del peso cargado no coincide con la esperada")
        param.data = val_tensor

    print(f"Modelo cargado correctamente desde '{archivo}'")
    return red


# =============================
# Notas finales
# =============================
# Este código busca mostrar cómo funcionan las redes neuronales a bajo nivel usando PyTorch.
# No utiliza nn.Module ni optimizadores de alto nivel para que el proceso matemático sea transparente.
# Las redes se pueden guardar y cargar fácilmente en formato JSON para su reutilización o análisis.
#
# Las funciones de activación y de pérdida utilizadas son las de torch y torch.nn.functional (F), que operan sobre tensores y permiten el cálculo automático de gradientes.
#
# Las clases están relacionadas jerárquicamente: MLP -> Layer
# Cada nivel encapsula el anterior, permitiendo construir redes de cualquier tamaño y profundidad.
