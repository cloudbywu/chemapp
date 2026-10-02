import torch
import torch.nn as nn
import numpy as np
import math
from torch import Tensor
from typing import Optional, Callable

class PositionalEncoding(nn.Module):

    """ Positional encoding with option of selecting specific indices to add """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 30000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: Tensor, ind: Tensor) -> Tensor:
        '''
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
            ind: Tensor, shape [batch_size, seq_len] or NoneType
        '''
        #Select and expand the PE to be the right shape first
        if ind is not None:
            added_pe = self.pe[torch.arange(1).reshape(-1, 1), ind, :]
            x = x + added_pe
        else:
            x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
    
class FlattenNNLinear(nn.Module):

    def __init__(self, 
                 d_model: int, 
                 d_out: int, 
                 d_feedforward: int,
                 max_seq_len: int):
        super().__init__()
        d1 = d_feedforward
        d2 = d1 * 2
        self.network = nn.Sequential(
            nn.Flatten(start_dim = 1),
            nn.Linear(d_model * max_seq_len, d1),
            nn.ReLU(),
            nn.Linear(d1, d2),
            nn.ReLU(),
            nn.Linear(d2, d2),
            nn.ReLU(),
            nn.Linear(d2, d_out),
            nn.Sigmoid()
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)
    
class NNLinearTransposeDownsize(nn.Module):

    def __init__(self,
                 d_model: int,
                 d_out: int,
                 d_feedforward: int,
                 max_seq_len: int,
                 n_layers: int,
                 layer_dimensions: list[int]):
        '''This module does not use d_feedforward: dimensions of the layers are determined
        by the layer_dimensions argument'''
        assert(n_layers == len(layer_dimensions))
        super().__init__()
        self.n_layers = n_layers
        self.layer_dims = layer_dimensions
        self.pretranspose = []
        for i in range(n_layers):
            if i == 0:
                self.pretranspose.append(nn.Linear(d_model, layer_dimensions[i]))
            else:
                self.pretranspose.append(nn.Linear(layer_dimensions[i - 1], layer_dimensions[i]))
            self.pretranspose.append(nn.ReLU())
        self.pretranspose = nn.Sequential(*self.pretranspose)
        self.posttranspose = nn.Sequential(
            nn.Linear(max_seq_len, d_out),
            nn.ReLU()
        )
        self.out = nn.Sequential(
            nn.Linear(layer_dimensions[-1], 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: Tensor) -> Tensor:
        '''
        x: Tensor, shape [batch_size, seq_len, d_model]
        '''
        x = self.pretranspose(x)
        #(N, T, D_n) -> (N, D_n, T)
        x = x.transpose(1, 2) 
        x = self.posttranspose(x)
        #(N, D_n, D_out) -> (N, D_out, D_n)
        x = x.transpose(1, 2) 
        x = self.out(x)
        return x.squeeze(-1)
    
class MHANet(nn.Module):

    model_id = 'MHANet'

    def __init__(self, 
                 src_embed: nn.Module,
                 positional_encoding: Optional[nn.Module],
                 forward_network: nn.Module, 
                 forward_network_opts: dict,
                 src_pad_token: int,
                 src_forward_function: Callable[[Tensor, nn.Module, int, Optional[nn.Module]], tuple[Tensor, Optional[Tensor]]], 
                 d_model: int,
                 d_out: int,
                 d_feedforward: int, 
                 n_heads: int, 
                 max_seq_len: int,
                 device: torch.device = None,
                 dtype: torch.dtype = None):
        """Implementation of MHA net which uses multihead attention

        Args:
            src_embed: The embedding module for the src tensor passed to the model
            positional_encoding: The positional encoding to use. Defaults to the PositionalEncoding class implemented
                in this module, but can be set to None to usee no positional encoding
            forward_network: The forward network that follows the MHA layer. Should take the d_model, d_out, d_feedforward, and max_seq_len arguments
                on __init__
            forward_network_opts: Dictionary of additional options to pass to the forward network. An empty dictionary is used to 
                indicate no additional arguments. 
            src_pad_token: The index used to indicate padding in the source sequence
            src_forward_function: A function that processes the src tensor using the src embedding, src pad token, and positional encoding to generate
                the embedded src and the src_key_pad_mask
            d_model: The size of the embedding dimension
            d_out: The size of the output dimension
            n_heads: The number of attention heads
            max_seq_len: The maximum sequence length
            device: The device to use for the model
            dtype: The datatype to use for the model
        """
        assert(d_model % n_heads == 0)
        super().__init__()
        self.src_embed = src_embed
        self.src_fwd_fn = src_forward_function
        self.src_pad_token = src_pad_token
        self.d_model = d_model
        self.d_out = d_out
        self.d_feedforward = d_feedforward
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype

        self.pos_encoder = (lambda x : x) if positional_encoding is None else positional_encoding(d_model)
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first = True)
        self.ffnn = forward_network(d_model, d_out, d_feedforward, max_seq_len, **forward_network_opts)

    def _sanitize_forward_args(self, x: tuple[Tensor, tuple[str]]) -> Tensor:
        #Unpack the tuple
        x, _ = x
        if isinstance(self.src_embed, nn.Embedding):
            x = x.long()
        return x

    def forward(self, x: tuple[Tensor, tuple[str]]) -> Tensor:
        src = self._sanitize_forward_args(x)
        src_embedded, src_key_pad_mask = self.src_fwd_fn(src, self.d_model, self.src_embed, self.src_pad_token, self.pos_encoder)
        x, _ = self.mha(src_embedded, 
                        src_embedded, 
                        src_embedded, 
                        key_padding_mask = src_key_pad_mask)
        return self.ffnn(x)
    
    def get_loss(self,
                 x: tuple[Tensor, tuple], 
                 y: tuple[Tensor], 
                 loss_fn: Callable[[Tensor, Tensor], Tensor]) -> Tensor:
        pred = self.forward(x)
        y_target, = y
        loss = loss_fn(pred, y_target.to(self.dtype).to(self.device))
        return loss