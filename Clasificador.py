import torch
import torch.nn.functional as F
from MLP import MLP, cargar_MLP, guardar_MLP
from Autoencoder import Autoencoder, cargar_autoencoder, guardar_autoencoder

class Clasificador:

    def __init__(self, 
                 encoder : MLP, 
                 n_classes : int,
                 estructura : list, 
                 f_act_salida : callable=None, 
                 f_act_oculta : callable=None):
        
        self.n_classes = n_classes
        self.dim_latente = len(encoder.layers[-1].neurons) # Igual es util que los mlp guarden el nin y noout de forma mas accesible
        
        self.mlp_clasificador = MLP(self.dim_latente,n_classes,estructura,f_act_salida,f_act_oculta)
        self.encoder = encoder
        
        self.layers = encoder.layers + self.mlp_clasificador.layers
    
    def parameters(self):
        return self.encoder.parameters() + self.mlp_clasificador.parameters()
    
    def __call__(self,x):
        """"""
        dimension_entrada = len(x)
        if not isinstance(x, torch.Tensor):
            raise TypeError("x debe ser un tensor de PyTorch")
        
        if dimension_entrada == self.dim_latente:
            classified = self.mlp_clasificador(x)

        else:
            encoded = self.encoder(x)
            classified = self.mlp_clasificador(encoded)
        
        return classified
    
    def train_classifier(self,
              training_encoded_data : list,
              target_vector : list,
              n_steps : int,
              stp_sz : float,
              loss_f : callable = F.cross_entropy,
              batch_size : int = None):
        
        if len(training_encoded_data) != len(target_vector):
            raise ValueError(f"Longitudes incompatibles: {len(training_encoded_data)} datos codificados y {len(target_vector)} etiquetas.")

        self.mlp_clasificador.train_model(training_encoded_data,target_vector,n_steps,stp_sz,loss_f,batch_size)
    
    def train_whole_model(self,
              training_encoded_data : list,
              target_vector : list,
              n_steps : int,
              stp_sz : float,
              loss_f : callable = F.cross_entropy,
              batch_size : int = None):
        ## NO IMPORTANTE, COMPLETAR MAS ADELANTE ##
        pass
        
def guardar_classificador(clasificador: Clasificador, archivo: str):
    """
    Guarda el estado de un Clasificador (encoder + MLP clasificador) en un archivo JSON.
    - clasificador: objeto Clasificador a guardar.
    - archivo: ruta del archivo destino.
    Se almacena:
      - estructura del encoder y del clasificador
      - funciones de activación de ambos
      - pesos de ambos componentes
    """
    state = [p.detach().tolist() for p in clasificador.parameters()]

    estructura_enc = [len(clasificador.encoder.layers[0].neurons[0].w)] + \
                     [len(layer.neurons) for layer in clasificador.encoder.layers]
    estructura_cls = [len(clasificador.mlp_clasificador.layers[0].neurons[0].w)] + \
                     [len(layer.neurons) for layer in clasificador.mlp_clasificador.layers]

    f_salida_enc = clasificador.encoder.f_act_salida.__name__ if clasificador.encoder.f_act_salida else None
    f_oculta_enc = clasificador.encoder.f_act_oculta.__name__ if clasificador.encoder.f_act_oculta else None
    f_salida_cls = clasificador.mlp_clasificador.f_act_salida.__name__ if clasificador.mlp_clasificador.f_act_salida else None
    f_oculta_cls = clasificador.mlp_clasificador.f_act_oculta.__name__ if clasificador.mlp_clasificador.f_act_oculta else None

    import json
    with open(archivo, "w") as f:
        json.dump({
            "estructura_encoder": estructura_enc,
            "estructura_clasificador": estructura_cls,
            "f_salida_encoder": f_salida_enc,
            "f_oculta_encoder": f_oculta_enc,
            "f_salida_clasificador": f_salida_cls,
            "f_oculta_clasificador": f_oculta_cls,
            "pesos": state
        }, f)
    print(f"Clasificador guardado en '{archivo}'\n")


def cargar_classificador(archivo: str) -> Clasificador:
    """
    Carga un Clasificador desde un archivo JSON.
    - archivo: ruta del archivo JSON con la configuración del clasificador.
    Devuelve un objeto Clasificador con el encoder y MLP de clasificación cargados.
    """
    import json
    with open(archivo, "r") as f:
        data = json.load(f)

    f_map = {
    "relu": F.relu,
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,
    "softmax": F.softmax,
    "log_softmax": F.log_softmax,
    "cross_entropy": F.cross_entropy,
    "identity": lambda x: x,  # esta sí puede quedarse como lambda si no usas su nombre
    None: None
}


    encoder = MLP(
        nin=data["estructura_encoder"][0],
        nout=data["estructura_encoder"][-1],
        estructura=data["estructura_encoder"][1:-1],
        f_act_salida=f_map[data["f_salida_encoder"]],
        f_act_oculta=f_map[data["f_oculta_encoder"]]
    )

    mlp_clasificador = MLP(
        nin=data["estructura_clasificador"][0],
        nout=data["estructura_clasificador"][-1],
        estructura=data["estructura_clasificador"][1:-1],
        f_act_salida=f_map[data["f_salida_clasificador"]],
        f_act_oculta=f_map[data["f_oculta_clasificador"]]
    )

    clasificador = Clasificador.__new__(Clasificador)  # Evita __init__

    clasificador.encoder = encoder
    clasificador.mlp_clasificador = mlp_clasificador
    clasificador.n_classes = data["estructura_clasificador"][-1]
    clasificador.dim_latente = len(encoder.layers[-1].neurons)
    clasificador.layers = encoder.layers + mlp_clasificador.layers

    # Y ahora sí, asigna los pesos después
    for p, w in zip(clasificador.parameters(), data["pesos"]):
        p.data = torch.tensor(w, dtype=torch.float32)
    print(f"Clasificador cargado correctamente desde '{archivo}'")
    return clasificador

