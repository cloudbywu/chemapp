from nmr.networks import mhanet, forward_fxns, embeddings
import torch
from torch import nn, Tensor
from typing import Callable, Optional, Any

class MHANetModel(nn.Module):

    model_id = 'MHANet'

    def __init__(self, 
                 src_embed: str,
                 src_embed_options: dict,
                 positional_encoding: Optional[str],
                 forward_network: str,
                 forward_network_opts: dict, 
                 src_pad_token: int,
                 src_forward_function: str,
                 source_size: int,
                 d_model: int,
                 d_out: int,
                 d_feedforward: int, 
                 n_heads: int, 
                 max_src_len: int,
                 freeze_components: Optional[list[str]] = None,
                 device: torch.device = None,
                 dtype: torch.dtype = None):
        """Implements a minimal multihead attention model

        Args:
            src_embed: String corresponding to the embedding module to use for the source sequence
            src_embed_options: Dictionary of additional arguments for the sourc embedding module
            positional_encoding: String corresponding to the positional encoding module to use. 
            forward_network: Name of the NN design to use after the MHA module
            forward_network_opts: Dictionary of additional options to pass to the forward network. Pass an 
                empty dictionary for no additional options.
            src_pad_token: The index used to indicate padding in the source sequence
            src_forward_function: Name of the function that processes the src tensor using the src embedding, src pad token, and positional encoding to generate
                the embedded src and the src_key_pad_mask
            source_size: The size of the source alphabet (including start, stop, and pad tokens)
            d_model: Embedding dimension of the model
            d_out: Output dimension of the model
            d_feedforward: Hidden dimension to use for feedforward networks in the forward network
            n_heads: Number of heads to use for multihead attention
            max_src_len: Maximum sequence length encountered in the dataset
            freeze_components: List of component names to freeze weights of
            device: The device to use for the model
            dtype: The datatype to use for the model
        """
        super().__init__()
        
        if src_embed is None:
            src_embed_module = None
        elif src_embed == 'nn.embed':
            src_embed_module = nn.Embedding(
                source_size, 
                d_model,
                padding_idx=src_pad_token,
                dtype=dtype,
                device=device
            )
        elif src_embed == 'spectra_continuous':
            #Add in options to src_embed_options to allow this embedding to handle triplets of points
            src_embed_module = embeddings.NMRContinuousEmbedding(d_model, **src_embed_options)
        elif src_embed == 'nn.embed_typed':
            src_embed_module = embeddings.NNEmbedWithTypeFeature(
                source_size,
                d_model,
                padding_idx = src_pad_token,
                **src_embed_options
            )
        else:
            raise ValueError("Unsupported source embedding")

        positional_encoding_module = getattr(mhanet, positional_encoding) if positional_encoding is not None else None
        forward_network_module = getattr(mhanet, forward_network)  
        src_forward_function = getattr(forward_fxns, src_forward_function)
        
        self.network = mhanet.MHANet(
            src_embed_module,
            positional_encoding_module,
            forward_network_module,
            forward_network_opts,
            src_pad_token,
            src_forward_function,
            d_model,
            d_out,
            d_feedforward,
            n_heads,
            max_src_len,
            device,
            dtype
        )

        self.initialize_weights()
        self.device = device
        self.dtype = dtype
        self.freeze_components = freeze_components

    def initialize_weights(self) -> None:
        """initialize network weights
        Non-1D parameters are initialized using Xavier initialization
        """
        for p in self.network.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

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
        return self.network(x)

    def get_loss(self, 
                 x: tuple[Tensor, tuple[str]], 
                 y: tuple[Tensor], 
                 loss_fn: Callable[[Tensor, Tensor], Tensor]) -> Tensor:
        return self.network.get_loss(x, y, loss_fn) 

