import torch
import torch.nn.functional as F
import random
from MLP import MLP, get_batches, guardar_MLP, cargar_MLP




"""

Fichero para probar posibles mejoras sin modificar el original que funciona correctamente.


IDEA: Es el momento de quitarse las listas y pasar a Vectorización del forward para batch (usando ventajas de pytorch) ?

"""

class Autoencoder:
    """
    Clase que representa un autoencoder.
    - encoder: red neuronal que codifica la entrada.
    - decoder: red neuronal que decodifica la representación comprimida.
    """
    def __init__(self, encoder : MLP, decoder : MLP):

        self.encoder = encoder
        self.decoder = decoder
        self.layers = encoder.layers + decoder.layers
    
    def parameters(self):
        return self.encoder.parameters() + self.decoder.parameters()

    def __call__(self,x):
        """
        Propaga la entrada x a través del codificador y decodificador.
        - Primero pasa por el codificador para obtener una representación comprimida.
        - Luego pasa por el decodificador para reconstruir la entrada original.
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def train_model(self, training_data: list, n_steps: int, step_sz: float,
                    loss_f: callable = F.mse_loss, batch_size: int = None,
                    beta: float = 1e-4, lambda_l2: float = 1e-4):
        """
        Entrena el autoencoder con:
        - Penalización L1 sobre la capa latente (sparsity).
        - Regularización L2 de pesos.
        
        Parámetros:
        - beta: peso de la penalización L1 sobre la representación latente.
        - lambda_l2: peso de la regularización L2 sobre los pesos.
        - eps: valor pequeño para evitar divisiones por cero en cálculos.
        """
        
        # Valores si recibimos None
        batch_size = len(training_data) if batch_size is None else batch_size
        beta = 0.0 if beta is None else beta
        lambda_l2 = 0.0 if lambda_l2 is None else lambda_l2
        

        parameters = self.parameters()

        for k in range(n_steps):
            epoch_loss = 0.0
            num_batches = 0

            for X_batch, Y_batch in get_batches(training_data, training_data, batch_size):
                # Gradientes a 0
                for p in parameters:
                    if p.grad is not None:
                        p.grad.zero_()

                # --- Forward ---
                encoded_batch = [self.encoder(x) for x in X_batch]    # Capa latente
                decoded_batch = [self.decoder(z) for z in encoded_batch]  # Reconstrucción

                # Pérdida de reconstrucción
                loss_recon = sum(loss_f(y_pred, y_true) 
                                for y_pred, y_true in zip(decoded_batch, Y_batch)) / len(Y_batch)

                # Penalización L1 sobre capa latente
                loss_l1 = sum(torch.mean(torch.abs(z)) for z in encoded_batch) / len(encoded_batch)

                # Regularización L2 sobre todos los parámetros
                loss_l2 = sum(torch.sum(p**2) for p in parameters)

                # Pérdida total
                loss = loss_recon + beta * loss_l1 + lambda_l2 * loss_l2

                # --- Backward ---
                loss.backward()

                # --- Actualización de parámetros ---
                for p in parameters:
                    p.data -= step_sz * p.grad

                epoch_loss += loss.item()
                num_batches += 1

            # Log
            if k % 20 == 0 or k == n_steps - 1:
                avg_loss = epoch_loss / num_batches
                print(f"Paso {k} | Loss total: {avg_loss:.6f} "
                    f"(Recon: {loss_recon.item():.6f}, L1: {loss_l1.item():.6f}, L2: {loss_l2.item():.6f})")


def guardar_autoencoder( autoencoder : Autoencoder, archivo : str):
    """
    Guarda el estado de un autoencoder en un archivo JSON.
    - autoencoder: objeto Autoencoder a guardar.
    - archivo: ruta del archivo destino.
    Se almacena:
      - estructura del codificador y decodificador
      - funciones de activación
      - pesos de ambos componentes
    """
    state = [p.detach().tolist() for p in autoencoder.parameters()]
    estructura_enc = [len(autoencoder.encoder.layers[0].neurons[0].w)] + [len(layer.neurons) for layer in autoencoder.encoder.layers]
    estructura_dec = [len(autoencoder.decoder.layers[0].neurons[0].w)] + [len(layer.neurons) for layer in autoencoder.decoder.layers]

    f_salida_enc = autoencoder.encoder.f_act_salida.__name__ if autoencoder.encoder.f_act_salida else None
    f_oculta_enc = autoencoder.encoder.f_act_oculta.__name__ if autoencoder.encoder.f_act_oculta else None
    f_salida_dec = autoencoder.decoder.f_act_salida.__name__ if autoencoder.decoder.f_act_salida else None
    f_oculta_dec = autoencoder.decoder.f_act_oculta.__name__ if autoencoder.decoder.f_act_oculta else None

    import json
    with open(archivo, "w") as f:
        json.dump({
            "estructura_encoder": estructura_enc,
            "estructura_decoder": estructura_dec,
            "f_salida_encoder": f_salida_enc,
            "f_oculta_encoder": f_oculta_enc,
            "f_salida_decoder": f_salida_dec,
            "f_oculta_decoder": f_oculta_dec,
            "pesos": state
        }, f)
    print(f"Autoencoder guardado en '{archivo}'\n")

def cargar_autoencoder(archivo : str):
    """
    Carga un autoencoder desde un archivo JSON.
    - archivo: ruta del archivo JSON con la configuración del autoencoder.
    Devuelve un objeto Autoencoder con el codificador y decodificador cargados.
    """
    import json
    with open(archivo, "r") as f:
        data = json.load(f)
    
    f_map = {"relu": F.relu, "sigmoid": torch.sigmoid, "tanh": torch.tanh, "softmax": lambda x: F.softmax(x, dim=0), None: None}

    encoder = MLP(
        nin=data["estructura_encoder"][0],
        nout=data["estructura_encoder"][-1],
        estructura=data["estructura_encoder"][1:-1],
        f_act_salida = f_map[data["f_salida_encoder"]],
        f_act_oculta=f_map[data["f_oculta_encoder"]]
    )

    decoder = MLP(
        nin=data["estructura_decoder"][0],
        nout=data["estructura_decoder"][-1],
        estructura=data["estructura_decoder"][1:-1],
        f_act_salida = f_map[data["f_salida_decoder"]],
        f_act_oculta=f_map[data["f_oculta_decoder"]]
    )

    for p, w in zip(encoder.parameters() + decoder.parameters(), data["pesos"]):
        p.data = torch.tensor(w, dtype=torch.float32)


    return Autoencoder(encoder, decoder)

