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
        #¿guardar n_clases y dim_latente como atributos?

        self.mlp_clasificador = MLP(self.dim_latente,n_classes,estructura,f_act_salida,f_act_oculta)
        self.encoder = encoder
        
        self.layers = encoder.layers + self.mlp_clasificador.layers
    
    def parameters(self):
        return self.encoder.parameters() + self.mlp_clasificador.parameters()
    
    def __call__(self,x):
        """"""
        dimension_entrada = len(x)
        
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
        # assert de dimension para que no se intente entrenar sin encoding?
        self.mlp_clasificador.train_model(training_encoded_data,target_vector,n_steps,stp_sz,loss_f,batch_size)
        
def guardar_classificador( clasificador : Clasificador, archivo : str):
    pass

def cargar_classificador( archivo : str):
    pass
