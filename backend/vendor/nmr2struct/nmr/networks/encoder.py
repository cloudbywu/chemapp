import numpy as np
import torch
from torch import nn, Tensor
import pickle, os
import math
import torch.nn.functional as f
from typing import Optional, Any, Callable

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

### Poolers ###
class TokenAvgPool(nn.Module):
    """Average pooling over the token dimension of the 3D encoder output"""
    def __init__(self):
        super().__init__()
    def forward(self, x: Tensor) -> Tensor:
        """Averages over the token dimension
        x: (N, T, E) -> (N, E) where 
            N = batch size
            T = sequence length
            E = feature dimension
        """
        return x.mean(dim=1)

class TokenMaxPool(nn.Module):
    """Max pooling over the token dimension of the 3D encoder output"""
    def __init__(self):
        super().__init__()
    def forward(self, x: Tensor) -> Tensor:
        """Takes the maximum value over the token dimensions
        x: (N, T, E) -> (N, E) where 
            N = batch size
            T = sequence length
            E = feature dimension
        """
        return x.max(dim=1).values
    
class SeqPool(nn.Module):
    """Sequence pooling operation introduced in https://arxiv.org/abs/2104.05704"""
    def __init__(self, d_model: int):
        super().__init__()
        self.g = nn.Linear(d_model, 1)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        x: (bsize, seq_len, d_model) -> (bsize, d_model)
        Double check this code: https://github.com/SHI-Labs/Compact-Transformers/blob/main/src/utils/transformers.py
        """
        x_p = f.softmax(self.g(x), dim = 1)
        z = torch.bmm(x_p.transpose(1, 2), x)
        return z.squeeze(1)

### Output heads ###
class SingleLinear(nn.Module):
    """A single linear layer with a sigmoid activation"""
    def __init__(self, d_model: int, d_out: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(d_model, d_out),
            nn.Sigmoid()
        )
    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)  
    
class EncoderNetwork(nn.Module):

    model_id = 'EncoderNet'

    def __init__(self, 
                 src_embed: nn.Module,
                 src_pad_token: int,
                 src_forward_function: Callable[[Tensor, nn.Module, int, Optional[nn.Module]], tuple[Tensor, Optional[Tensor]]],
                 pooler: nn.Module,
                 pooler_opts: dict, 
                 output_head: nn.Module,
                 output_head_opts: dict,
                 d_model: int = 512,
                 nhead: int = 8,
                 dim_feedforward: int = 2048,
                 d_out: int = 957,
                 source_size: int = 957,
                 dropout: float = 0.1,
                 activation: str = 'relu',
                 layer_norm_eps: float = 1e-05,
                 batch_first: bool = True,
                 enable_norm: bool = True,
                 norm_first: bool = False,
                 bias: bool = True,
                 num_layers: int = 6,
                 enable_nested_tensor: bool = True,
                 device: torch.device = None,
                 dtype: torch.dtype = torch.float
                 ):
        r"""Most parameters are for the PyTorch TransformerEncoderLayer module.
        Args:
            src_embed: The embedding module for the src tensor passed to the model
            src_pad_token: The index used to indicate padding in the source sequence
            src_forward_function: A function that processes the src tensor using the src embedding, src pad token, and positional encoding to generate
                the embedded src and the src_key_pad_mask
            pooler: Module for pooling the output of the transformer. This is typically seen in sequence classification 
                tasks where pooling is applied in the sequence length/time dimension, i.e.:
                (N, T, E) --> (N, E) where N = batch size, T = seq len, and E = feature dimension.
                The pooler should take in a 3D input of shape (N, T, E) and produce an output of shape (N, E)
            pooler_opts: Additional options for the pooling head
            output_head: Module for generating the output from the pooled transformer output. 
                This module should take at minimum a 2D input of shape (N, E) and produce an output of shape (N, d_out)
            output_head_opts: Additional options for the output head
            d_model: The dimensionality of the model
            nhead: The number of heads in the multiheadattention models
            dim_feedforward: The inner dimension of the feedforward network model
            d_out: The dimensionality of the output (e.g. 957 for the number of substructures)
            source_size: The size of the source vocabulary, 
            batch_first: If True, then all tensors are shaped as (N, T, E), with N being batch size
            enable_norm: If True, then layer normalization is applied as in the original transformer encoder + decoder.
                Default is True
            num_layers: The number of layers to use in the encoder
            enable_nested_tensor: Whether nested tensor operations are enabled inside the transformer. By default enabled,
                improves performance when padding is high
            device: The device for the model
            dtype: The dtype for the model
        """
        super().__init__()
        self.src_embed = src_embed
        self.src_size = source_size
        self.src_fwd_fn = src_forward_function
        self.src_pad_token = src_pad_token
        self.d_model = d_model
        self.d_out = d_out
        self.nhead = nhead
        self.nlayers = num_layers

        self.pooler = pooler(**pooler_opts)
        self.output_head = output_head(**output_head_opts)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        self.dtype = dtype
        self.device = device

        #Construct the encoder model. Process taken from 
        #   https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
            bias=bias,
            device=device,
            dtype=dtype
        )
        if enable_norm:
            norm = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, device=device, dtype=dtype)
        else:
            norm = None
        self.encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers, 
            norm=norm,
            enable_nested_tensor=enable_nested_tensor
        )
    
    def _sanitize_forward_args(self, x: tuple[Tensor, tuple[str]]) -> Tensor:
        x, _ = x
        if isinstance(self.src_embed, nn.Embedding):
            x = x.long()
        return x
    
    def forward(self, x: tuple[Tensor, tuple[str]]) -> Tensor:
        src = self._sanitize_forward_args(x)
        src_embedded, src_key_pad_mask = self.src_fwd_fn(src, self.d_model, self.src_embed, self.src_pad_token, self.pos_encoder)
        src_out = self.encoder(src=src_embedded,
                               src_key_padding_mask=src_key_pad_mask)
        src_out = self.pooler(src_out)
        return self.output_head(src_out)
    
    def get_loss(self, 
                 x: tuple[Tensor, tuple[str]],
                 y: tuple[Tensor],
                 loss_fn: Callable[[Tensor, Tensor], Tensor]) -> Tensor:
        pred = self.forward(x)
        y_target, = y
        loss = loss_fn(pred, y_target.to(self.dtype).to(self.device))
        return loss