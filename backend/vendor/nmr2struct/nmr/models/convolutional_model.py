from nmr.networks import convolutional
import torch
from torch import nn, Tensor
from typing import Tuple, Callable, Optional

class ConvolutionalModel(nn.Module):
    """ Example model wrapper for convolutional network """

    def __init__(self, 
                 n_spectral_features: int, 
                 n_Cfeatures: int, 
                 n_molfeatures: int, 
                 n_substructures: int,
                 freeze_components: Optional[list] = None,
                 device: torch.device = None, 
                 dtype: torch.dtype = torch.float):
        """Constructor for convolutional network model using the NMRConvNet from networks
        Args:
            n_spectral_features: The number of spectral features, i.e. 28000
            n_Cfeatures: The number of CNMR features, i.e. 40
            n_molfeatures: The number of chemical formula features, i.e. 5
            n_substructures: The number of substructures to predict for. This is used for 
                constructing a single linear head for each substructure
            freeze_components: List of component names to freeze
            device: Model device. Default is None
            dtype: Model datatype. Default is torch.float
        """
        super().__init__()
        self.network = convolutional.NMRConvNet(n_spectral_features,
                                                n_Cfeatures,
                                                n_molfeatures,
                                                n_substructures,
                                                dtype,
                                                device)
        self.initialize_weights()
        self.freeze_components = freeze_components
        self.device = device
        self.dtype = dtype

    def initialize_weights(self) -> None:
        """initialize network weights
        Non-1D parameters are initialized using Xavier initialization,
        1D parameters are initialized to 0
        """
        for p in self.network.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif p.dim() == 1:
                nn.init.zeros_(p)

    def freeze(self) -> None:
        """Disables gradients for specific components of the network
        
        Args:
            components: A list of strings corresponding to the model components
                to freeze, e.g. src_embed, tgt_embed.
        """
        #TODO: This will need careful testing
        if self.freeze_components is not None:
            for component in self.freeze_components:
                if hasattr(self.network, component):
                    for param in getattr(self.network, component).parameters():
                        param.requires_grad = False
    
    def forward(self, x: tuple[Tensor, tuple[str]]) -> Tensor:
        """
        Args:
            x: ((batch_size, 1, seq_len), smiles)
        """
        return self.network(x)
    
    def get_loss(self,
                 x: Tuple[Tensor, Tuple], 
                 y: Tuple[Tensor], 
                 loss_fn: Callable[[Tensor, Tensor], Tensor]) -> Tensor:
        return self.network.get_loss(x, y, loss_fn)
        



