import nmr.models
import torch
import torch.nn as nn
from typing import Any
import warnings
# warnings.simplefilter('always', UserWarning)

def create_model(model_args: dict, dtype: torch.dtype, device: torch.device) -> nn.Module:
    """Creates the model by passing argument dictoinaries into fetched constructors
    Args:
        model_args: The dictionary of model arguments, possibly a highly nested structure
        dtype: The datatype to use for the model
        device: The device to use for the model
    """
    model_base = getattr(nmr.models, model_args['model_type'])
    model_config = model_args['model_args']
    model = model_base(dtype=dtype,
                       device=device,
                       **model_config)
    if model_args['load_model'] is not None:
        ckpt = torch.load(model_args['load_model'], map_location=device)['model_state_dict']
        try:
            model.load_state_dict(ckpt)
            print("Model loaded successfully")
        except Exception as e:
            print(e)
            warnings.warn("Keys do not match, so loading partial weights where possible")
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

    #Freeze requisite components
    model.freeze()

    #Update the model args
    model_args['model_args'] = model_config

    return model, model_args
