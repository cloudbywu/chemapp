from nmr.networks import transformer, forward_fxns, embeddings
import torch
from torch import nn, Tensor
from typing import Tuple, Callable, Optional, Any
import nmr.models
import warnings

class MultiTaskModel(nn.Module):
    """ Model that predicts substructures and structures """

    def __init__(self, 
                 src_embed: str,
                 src_embed_options: dict,
                 structure_model: str,
                 structure_model_args: dict,
                 substructure_model: str,  
                 substructure_model_args: dict,
                 forward_fxn: str, 
                 structure_model_ckpt: str,
                 substructure_model_ckpt: str,
                 device: torch.device = None,
                 dtype: torch.dtype = torch.float):
        """Constructor for multitask model that takes in a src embedding used to produce a structure and substructure prediction"""
        super().__init__()
        self.src_embed = getattr(embeddings, src_embed)(**src_embed_options)
        self.structure_model = getattr(nmr.models, structure_model)(
            **structure_model_args
        )
        self.substructure_model = getattr(nmr.models, substructure_model)(
            **substructure_model_args
        )
        src_embed_dim = self.src_embed.d_model
        structure_model_dim = self.structure_model.network.d_model
        substructure_model_dim = self.substructure_model.network.d_model
        
        #Connecting linear transformations
        self.structure_connector = nn.Linear(src_embed_dim, structure_model_dim)
        self.substructure_connector = nn.Linear(src_embed_dim, substructure_model_dim)

        self.fwd_fn = getattr(forward_fxns, forward_fxn)
        self.device = device
        self.dtype = dtype
        self.initialize_weights()
        if structure_model_ckpt is not None:
            self._partial_load_weights(self.structure_model, structure_model_ckpt)
        if substructure_model_ckpt is not None:
            self._partial_load_weights(self.substructure_model, substructure_model_ckpt)

    def initialize_weights(self) -> None:
        """initialize network weights"""
        self.structure_model.initialize_weights()
        self.substructure_model.initialize_weights()
    
    def freeze(self) -> None:
        """Disables gradients for specific components of the network"""
        self.structure_model.freeze()
        self.substructure_model.freeze()

    def _partial_load_weights(self, model: nn.Module, ckpt: str) -> None:
        ckpt = torch.load(ckpt, map_location = self.device)['model_state_dict']
        model_state = model.state_dict()
        pretrained_dictionary = {}
        for k, v in ckpt.items():
            if k in model_state:
                if model_state[k].shape == v.shape:
                    pretrained_dictionary[k] = v
                else:
                    warnings.warn(f"Could not load {k}: expected {model_state[k].shape} but got {v.shape}")
            else:
                warnings.warn(f"Could not load {k} because it is not in the model state dictionary")
        print("The following keys are ignored in the model:")
        for k in model_state:
            if k not in pretrained_dictionary:
                print(k)
        model.load_state_dict(pretrained_dictionary, strict=False)
    
    def _sanitize_forward_args(self, x, y):
        inp, _ = x
        structure_targets, substructure_targets = y
        return inp, structure_targets, substructure_targets
    
    def _unpack_to_list(self, x: Tensor, dim: int) -> list[Tensor]:
        """
        Given a tensor of elements, unpacks the tensor into a tuple of tensors. For exasmple,
        a tensor of shape (N, 2, E) -> ((N, E), (N, E))
        """
        return [torch.select(x, dim, i) for i in range(x.shape[dim])]

    def forward(self, 
                x: Tuple[Tensor, Tuple], 
                y: Tuple[Tensor, Tensor],
                eval_paths: list[str]) -> Tensor:
        """
        The argument eval_paths indicates which submodels to evaluate. It can contain the following values:
            'structure': The structure model is evaluated on this forward pass
            'substructure': The substructure model is evaluated on this forward pass
        Note that the forward() function is not used in get_loss. This is because get_loss will always 
        evaluate both submodels. If one wishes to only use one submodel, they should train the models 
        using the convolutional embedding instead. 
        """
        src, struct_targs, substruct_targs = self._sanitize_forward_args(x, y)
        if 'structure' in eval_paths:
            src_struct_embedded, src_struct_key_pad_mask = self.fwd_fn(src,
                                                                   self.structure_model.network.d_model, 
                                                                   self.src_embed, 
                                                                   self.structure_model.network.src_pad_token, 
                                                                   self.structure_model.network.pos_encoder)
            src_struct_embedded = self.structure_connector(src_struct_embedded)
            structure_output = self.structure_model(
                ((src_struct_embedded, src_struct_key_pad_mask), None),
                struct_targs
            )
        else:
            structure_output = None
        if 'substructure' in eval_paths:
            src_substruct_embedded, src_substruct_key_pad_mask = self.fwd_fn(src,
                                                                        self.substructure_model.network.d_model, 
                                                                        self.src_embed, 
                                                                        self.substructure_model.network.src_pad_token, 
                                                                        self.substructure_model.network.pos_encoder)
            
            src_substruct_embedded = self.substructure_connector(src_substruct_embedded)
            
            substructure_output = self.substructure_model(
                ((src_substruct_embedded, src_substruct_key_pad_mask), None)
            )
        else:
            substructure_output = None
        return structure_output, substructure_output
    
    def get_loss(self,
                 x: Tuple[Tensor, Tuple], 
                 y: Tuple[Tensor], 
                 loss_fn: Callable[[Tensor, Tensor], Tensor]) -> Tensor:
        structure_loss = lambda x, y : loss_fn('structure', x, y)
        substructure_loss = lambda x, y : loss_fn('substructure', x, y)
        src, struct_targs, substruct_targs = self._sanitize_forward_args(x, y)
        src_struct_embedded, src_struct_key_pad_mask = self.fwd_fn(src,
                                                                   self.structure_model.network.d_model, 
                                                                   self.src_embed, 
                                                                   self.structure_model.network.src_pad_token, 
                                                                   self.structure_model.network.pos_encoder)
        src_substruct_embedded, src_substruct_key_pad_mask = self.fwd_fn(src,
                                                                       self.substructure_model.network.d_model, 
                                                                       self.src_embed, 
                                                                       self.substructure_model.network.src_pad_token, 
                                                                       self.substructure_model.network.pos_encoder)
        src_struct_embedded = self.structure_connector(src_struct_embedded)
        src_substruct_embedded = self.substructure_connector(src_substruct_embedded)
        # import pdb; pdb.set_trace()
        #Loss scaling factors are multiplied within forward() calls
        structure_loss = self.structure_model.get_loss(((src_struct_embedded, src_struct_key_pad_mask), None), 
                                                       self._unpack_to_list(struct_targs, 1), 
                                                       structure_loss)
        substructure_loss = self.substructure_model.get_loss(((src_substruct_embedded, src_substruct_key_pad_mask), None),
                                                              (substruct_targs,), substructure_loss)
        return structure_loss + substructure_loss