import torch
import torch.nn.functional as F
from MLP import MLP, cargar_MLP, guardar_MLP, nombre_a_func
from Autoencoder import Autoencoder, cargar_autoencoder, guardar_autoencoder

## Funciones relevantes ##

def onehot_to_long(targets: torch.Tensor) -> torch.Tensor:
    """
    Para cuando se use F.cross_entropy.
    Detecta si los targets están en formato one-hot y los convierte a índices de clase (long).
    Si ya están en formato entero, los deja tal cual.
    """
    if not isinstance(targets, torch.Tensor):
        targets = torch.stack(targets)

    # Verifica si es un tensor 2D y si cada fila tiene una única posición con valor 1
    if targets.ndim == 2 and torch.all((targets.sum(dim=1) == 1)) and torch.all((targets == 0) | (targets == 1)):
        # Es one-hot → convertir a índices
        print("Los targets estaban en forma one-hot y se han transformado a long")
        return torch.argmax(targets, dim=1).long()
    else:
        # Ya está en formato correcto o no es one-hot
        return targets.long()


class Clasificador:

    def __init__(self, 
                 encoder : MLP, 
                 n_classes : int,
                 estr_oc_clas : list,
                 list_act_clas: list):
        
        self.n_classes = n_classes
        self.dim_latente = encoder.dim_out
        self.mlp_clasificador = MLP(self.dim_latente, n_classes, estr_oc_clas, list_act_clas)
        self.encoder = encoder
        self.layers = encoder.layers + self.mlp_clasificador.layers
    
    def parameters(self):
        return self.encoder.parameters() + self.mlp_clasificador.parameters()
    
    def __call__(self,x:torch.Tensor):
        """"""
        if not isinstance(x, torch.Tensor):
            raise TypeError("x debe ser un tensor de PyTorch")
        
        if x.shape[-1] == self.dim_latente: 
            print("El clasificador ha interpretado que x ya estaba codificado.\n")
            classified = self.mlp_clasificador(x)

        elif x.shape[-1] == self.encoder.dim_in:
            encoded = self.encoder(x)
            classified = self.mlp_clasificador(encoded)
        else:
            raise ValueError(
                f"Dimensión de entrada no válida: {x.shape[-1]}. "
                f"Esperado {self.encoder.dim_in} (datos crudos) o {self.dim_latente} (codificados)."
            )
        
        return classified
    
    def train_classifier(self,
              training_data : list[torch.Tensor],
              target_vector : list[torch.Tensor],
              n_steps : int,
              stp_sz : float,
              loss_f : callable = F.cross_entropy,
              batch_size : int = None):
            
        
        if not isinstance(training_data, torch.Tensor):
            training_data = torch.stack(training_data)
        
        # Si la funcion de perdida es Cross entropy, retiramos temporalmente la ultima activacion pq torch ya le aplica softmax
        if loss_f == F.cross_entropy:
            last_act = self.mlp_clasificador.activaciones[-1]
            self.mlp_clasificador.activaciones[-1] = None
            target_vector = onehot_to_long(target_vector)


        if training_data[0].shape[-1] == self.dim_latente :
            self.mlp_clasificador.train_model(training_data, target_vector, n_steps, stp_sz, loss_f, batch_size)
        else:
            with torch.no_grad():
                training_encoded_data = self.encoder(training_data)
            self.mlp_clasificador.train_model(training_encoded_data, target_vector, n_steps, stp_sz, loss_f, batch_size)
        
        if loss_f == F.cross_entropy:
            self.mlp_clasificador.activaciones[-1] = last_act
    

    def train_whole_model(self,
              training_data : list[torch.Tensor],
              target_vector : list[torch.Tensor],
              n_steps : int,
              stp_sz : float,
              loss_f : callable = F.cross_entropy,
              batch_size : int = None):
        ## NO IMPORTANTE, COMPLETAR MAS ADELANTE ##
        print("Aun no esta operativo. No se ha entrenado el modelo")
        
        
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

    estructura_enc = clasificador.encoder.dims
    estructura_cls = clasificador.mlp_clasificador.dims

    activaciones_enc = [ getattr(f_act, "__name__", "none") if f_act else "none" for f_act in clasificador.encoder.activaciones] 
    activaciones_mlp_clas = [ getattr(f_act, "__name__", "none") if f_act else "none" for f_act in clasificador.mlp_clasificador.activaciones] 

    import json
    with open(archivo, "w") as f:
        json.dump({
            "tipo_modelo": "clasificador",
            "estructura_encoder": estructura_enc,
            "estructura_clasificador": estructura_cls,
            "activaciones_enc": activaciones_enc,
            "activaciones_mlp_clas": activaciones_mlp_clas,
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


    encoder = MLP(
        dim_in=data["estructura_encoder"][0],
        dim_out=data["estructura_encoder"][-1],
        estructura_oct=data["estructura_encoder"][1:-1],
        f_act_list=[nombre_a_func.get(f, None) for f in data["activaciones_enc"]]
    )

    mlp_clasificador = MLP(
        dim_in=data["estructura_clasificador"][0],
        dim_out=data["estructura_clasificador"][-1],
        estructura_oct=data["estructura_clasificador"][1:-1],
        f_act_list=[nombre_a_func.get(f, None) for f in data["activaciones_mlp_clas"]]
    )

    clasificador = Clasificador.__new__(Clasificador)  # Evita __init__

    clasificador.encoder = encoder
    clasificador.mlp_clasificador = mlp_clasificador
    clasificador.n_classes = mlp_clasificador.dim_out
    clasificador.dim_latente = encoder.dim_out
    clasificador.layers = encoder.layers + mlp_clasificador.layers

    # Y ahora sí, asigna los pesos después
    for p, w in zip(clasificador.parameters(), data["pesos"]):
        p.data = torch.tensor(w, dtype=torch.float32)
    print(f"Clasificador cargado correctamente desde '{archivo}'")
    return clasificador

