import torch
from torch import nn, Tensor
from typing import Tuple, Optional, Callable
import math

### Target foward processing functions ###

def tgt_fwd_fxn_basic(tgt: Tensor,
                      d_model: int, 
                      tgt_embed: nn.Module,
                      tgt_pad_token: int,
                      pos_encoder: nn.Module) -> Tuple[Tensor, Optional[Tensor]]:
    """Standard forward processing function for target tensor used in the Transformer network 
    Args:
        tgt: The unembedded target tensor, raw input into the forward() method,
            shape (batch_size, seq_len)
        d_model: The dimensionality of the model
        tgt_embed: The target embedding layer
        tgt_pad_token: The target padding token index
        pos_encoder: The positional encoder layer
    """
    if tgt_pad_token is not None:
        tgt_key_pad_mask = (tgt == tgt_pad_token).bool().to(tgt.device)
    else:
        tgt_key_pad_mask = None
    
    tgt = tgt_embed(tgt) * math.sqrt(d_model)
    tgt = pos_encoder(tgt, None)
    return tgt, tgt_key_pad_mask

### Source foward processing functions ###

def src_fwd_fxn_basic(src: Tensor,
                      d_model: int,
                      src_embed: nn.Module,
                      src_pad_token: int,
                      pos_encoder: nn.Module) -> Tuple[Tensor, Optional[Tensor]]:
    """Forward processing for source tensor in Transformer for substructure to structure problem
    Args:
        src: The unembedded source tensor, raw input into the forward() method, 
            shape (batch_size, seq_len)
        d_model: The dimensionality of the model
        src_embed: The source embedding layer
        src_pad_token: The source padding token index
        pos_encoder: The positional encoder layer
    """
    if not isinstance(src_embed, nn.Embedding):
        src_key_pad_mask = None
    elif src_pad_token is not None:
        src_key_pad_mask = (src == src_pad_token).bool().to(src.device)
    src = src_embed(src) * math.sqrt(d_model)
    src = pos_encoder(src, None)
    return src, src_key_pad_mask

def src_fwd_fxn_conv_embedding(src: Tensor,
                               d_model: int,
                               src_embed: nn.Module,
                               src_pad_token: int,
                               pos_encoder: nn.Module) -> Tuple[Tensor, Optional[Tensor]]:
    """Forward processing for the source tensor where the input is an unprocessed spectrum
    This forward function is only to be used with the convolutional source embedding. It has hard-coded values
    to allow for all shapes to line up with the current spectrum representation
    Args:
        src: The unembedded source tensor, in this case representing a spectrum, (batch_size, seq_len)
        d_model: The dimensionality of the model
        src_embed: The source embedding layer
        src_pad_token: The source padding token index. In the case of this forward function, it is 
            hard-coded to 0
        pos_encoder: The positional encoder layer
    """
    assert(src_embed is not None)
    #Only construct cnmr padding mask if using cnmr information
    if src_embed.use_cnmr:
        cnmr_start = src_embed.n_spectral_features
        cnmr_end = cnmr_start + src_embed.n_Cfeatures
        cnmr = src[:, cnmr_start:cnmr_end]
        assert(cnmr.shape[-1] == src_embed.n_Cfeatures)
        sorted_cnmr = torch.sort(cnmr, dim = -1, descending=True).values
        cnmr_key_pad_mask = (sorted_cnmr == 0).bool().to(src.device)
    else:
        cnmr_key_pad_mask = torch.tensor([]).bool().to(src.device)

    if src_embed.use_hnmr:
        src_key_pad_mask = torch.zeros(src.shape[0], 
                                    src_embed.h_spectrum_final_seq_len).bool().to(src.device)
    else:
        src_key_pad_mask = torch.tensor([]).bool().to(src.device)
    
    #Concatenate the padding masks together
    src_key_pad_mask = torch.cat((src_key_pad_mask, cnmr_key_pad_mask), dim = -1)
    
    src_embedded = src_embed(src) * math.sqrt(d_model)
    if src_embed.add_pos_encoder:
        src_embedded = pos_encoder(src_embedded, None)
    return src_embedded, src_key_pad_mask

def src_fwd_fxn_spectra_tokenized(src: Tensor,
                                  d_model: int,
                                  src_embed: nn.Module,
                                  src_pad_token: int,
                                  pos_encoder: nn.Module) -> Tuple[Tensor, Optional[Tensor]]:
    """Forward processing for source tensor in Transformer + MHANet with tokenized spectra
    Args:
        src: The unembedded source tensor, raw input into the forward() method, shape 
            (batch_size, 2, seq_len)
        d_model: The dimensionality of the model
        src_embed: The source embedding layer
        src_pad_token: The source padding token index
        pos_encoder: The positional encoder layer
    """
    assert(src.shape[1] == 2)
    src = src.long()
    src_unembedded, src_inds = src[:,0,:], src[:,1,:]
    src_embedded = src_embed(src_unembedded) * math.sqrt(d_model)
    src_embedded = pos_encoder(src_embedded, src_inds)
    src_key_pad_mask = (src_unembedded == src_pad_token).bool().to(src_unembedded.device)
    return src_embedded, src_key_pad_mask

def src_fwd_fxn_spectra_tokenized_with_type(src: Tensor,
                                            d_model: int,
                                            src_embed: nn.Module,
                                            src_pad_token: int,
                                            pos_encoder: nn.Module) -> Tuple[Tensor, Optional[Tensor]]:
    """Forward processing for source tensor in Transformer + MHANet with tokenized spectra and type information
    Args:
        src: The unembedded source tensor, raw input into the forward() method, shape 
            (batch_size, 3, seq_len)
        d_model: The dimensionality of the model
        src_embed: The source embedding layer
        src_pad_token: The source padding token index
        pos_encoder: The positional encoder layer
    """
    assert(src.shape[1] == 3)
    src = src.long()
    src_inds = src[:,1,:]
    src_embedded = src_embed(src) * math.sqrt(d_model)
    src_embedded = pos_encoder(src_embedded, src_inds)
    src_key_pad_mask = (src[:,0,:] == src_pad_token).bool().to(src.device)
    return src_embedded, src_key_pad_mask

def src_fwd_fxn_spectra_continuous(src: Tensor, 
                                   d_model: int,
                                   src_embed: nn.Module,
                                   src_pad_token: int,
                                   pos_encoder: nn.Module) -> Tuple[Tensor, Optional[Tensor]]:
    """Forward processing for source tensor in Transformer + MHANet with continuous spectra pairs
    Args:
        src: The unembedded source tensor, raw input into the forward() method, shape 
            (batch_size, seq_len, 2)
        d_model: The dimensionality of the model
        src_embed: The source embedding layer
        src_pad_token: The source padding token index
        pos_encoder: The positional encoder layer
    """
    assert(src.shape[2] == 2 or src.shape[2] == 3)
    src_key_pad_mask = (src[:,:,0] == src_pad_token).bool().to(src.device)
    src_embedded = src_embed(src) * math.sqrt(d_model)
    src_embedded = pos_encoder(src_embedded, None)
    return src_embedded, src_key_pad_mask

def src_fwd_fxn_no_embedding_mlp(src: Tensor, 
                                 d_model: int,
                                 src_embed: nn.Module,
                                 src_pad_token: int,
                                 pos_encoder: nn.Module) -> Tuple[Tensor, Optional[Tensor]]:
    """Forward processing for source tensor in Transformer + MHANet with no embedding MLP before the positional encoding
    Args:
        src: The unembedded source tensor, raw input into the forward() method, shape 
            (batch_size, seq_len, 2)
        d_model: The dimensionality of the model
        src_embed: The source embedding layer
        src_pad_token: The source padding token index
        pos_encoder: The positional encoder layer
    """
    assert(src.shape[2] == 2)
    assert(src_embed is None)
    src_key_pad_mask = (src[:,:,0] == src_pad_token).bool().to(src.device)
    src_embedded = pos_encoder(src, None)
    return src_embedded, src_key_pad_mask

def src_fwd_fxn_packed_tensor(src: tuple[Tensor],
                              d_model: int,
                              src_embed: nn.Module,
                              src_pad_token: int,
                              pos_encoder: nn.Module) -> Tuple[Tensor, Optional[Tensor]]:
    """ Forward processing for a source tensor that is a tuple which contains 
    the embedded sequence and the padding mask"""
    assert(src_embed is None)
    src_embedded, src_key_pad_mask = src
    return src_embedded, src_key_pad_mask

### Forward functions for the combined model ###
def mod1_x_expand_dim_mod2_xy(model_1: nn.Module,
                              model_2: nn.Module,
                              x: tuple[Tensor, tuple[str]],
                              y: tuple[Tensor, Tensor] | tuple[Tensor]) -> Tensor:
    _, smiles = x
    mod1_output = model_1(x)
    mod1_output = mod1_output.unsqueeze(-1)
    final_output = model_2((mod1_output, smiles), y)
    return final_output
