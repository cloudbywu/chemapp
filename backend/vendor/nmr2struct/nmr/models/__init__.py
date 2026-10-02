from .transformer_model import TransformerModel
from .convolutional_model import ConvolutionalModel
from .combined_model import CombinedModel
from .mhanet_model import MHANetModel
from .encoder_model import EncoderModel
from .multitask_model import MultiTaskModel
from .build_model import create_model

def get_all_models():
    all_models = [
        "TransformerModel",
        "ConvolutionalModel",
        "CombinedModel"
    ]
    print(all_models)