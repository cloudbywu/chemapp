import os

if os.environ.get("NFP_NO_KERAS") != "1":
    from .layers import *
    from .models import *

    custom_layers = {
        'MessageLayer': MessageLayer,
        'EdgeNetwork': EdgeNetwork,
        'ReduceAtomToMol': ReduceAtomToMol,
        'ReduceBondToAtom': ReduceBondToAtom,
        'GatherAtomToBond': GatherAtomToBond,
        'GRUStep': GRUStep,
        'Embedding2D': Embedding2D,
        'Squeeze': Squeeze,
        'GraphModel': GraphModel,
        'masked_mean_squared_error': masked_mean_squared_error
    }
else:
    custom_layers = {}
