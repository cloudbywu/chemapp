import torch
from torch import nn, Tensor
from typing import Callable, Optional
import nmr.models
from nmr.networks import forward_fxns

class CombinedModel(nn.Module):
    """Example model wrapper for the combined model"""
    model_id = 'combined'

    def __init__(self, model_1: str, 
                 model_2: str, 
                 model_1_args: dict, 
                 model_2_args: dict,
                 forward_fxn: str,
                 model_1_ckpt: str = None, 
                 model_2_ckpt: str = None,
                 device: torch.device = None, 
                 dtype: torch.dtype = torch.float):
        """Constructor for combined model built from two sub models
        
        Args:
            model_1: The name of the first sub model
            model_2: The name of the second sub model
            model_1_args: The kwargs for the first sub model constructor
            model_2_args: The kwargs for the second sub model constructor
            forward_fxn: A function which takes two models, the input x, y, and returns the output
            model_1_ckpt: The checkpoint to load for model 1
            model_2_ckpt: The checkpoint to load for model 2
            device: Model device. Default is None
            dtype: Model datatype. Default is torch.float
        """
        super().__init__()
        self.model_1 = getattr(nmr.models, model_1)(
            **model_1_args
        )
        self.model_2 = getattr(nmr.models, model_2)(
            **model_2_args
        )
        self.fwd_fn = getattr(forward_fxns, forward_fxn)
        self.device = device
        self.dtype = dtype
        self.initialize_weights()
        #Initialize the weights for each sub checkpoint if not loading the 
        #   overall model state dictionary
        if model_1_ckpt is not None:
            ckpt = torch.load(model_1_ckpt, map_location=device)
            self.model_1.load_state_dict(ckpt['model_state_dict'])
        if model_2_ckpt is not None:
            ckpt = torch.load(model_2_ckpt, map_location=device)
            self.model_2.load_state_dict(ckpt['model_state_dict'])
    
    def initialize_weights(self) -> None:
        """initialize network weights"""
        self.model_1.initialize_weights()
        self.model_2.initialize_weights()

    def freeze(self) -> None:
        """Disables gradients for specific components of the network
        
        Args:
            components: A list of strings corresponding to the model components
                to freeze, e.g. src_embed, tgt_embed.

        Since the combined model uses two other models, call each model's respective
        freeze method. The components have already been stored as class attributes
        """
        self.model_1.freeze()
        self.model_2.freeze()
    
    #TODO: rework this forward function
    def forward(self, 
                x: tuple[Tensor, tuple[str]],
                y: tuple[Tensor, Tensor] | tuple[Tensor]) -> Tensor:
        '''
        Args:
            x: ((batch_size, 1, seq_len), smiles)
            y: ((batch_size, seq_len), (batch_size, seq_len)) or ((batch_size, seq_len),)
        '''
        return self.fwd_fn(self.model_1, self.model_2, x, y)
    
    def get_loss(self, 
                 x: tuple[Tensor, tuple],
                 y: tuple[Tensor, Tensor] | tuple[Tensor],
                 loss_fn: Callable[[Tensor, Tensor], Tensor]) -> Tensor:
        pred = self.forward(x, y)
        if pred.dim() == 3:
            pred = pred.permute(0, 2, 1)
            tgt = y[1].to(self.device)
        else:
            tgt = y[0].to(self.device)
        if isinstance(loss_fn, torch.nn.CrossEntropyLoss):
            tgt = tgt.long()
        loss = loss_fn(pred, tgt)
        return loss