"""ForwardGNN-13C prototype: molecule graph -> atom-level 13C shift prediction.

This package is the reference implementation of the forward module described
in docs/optimal-ml-module-design-v1.md.  It is intentionally self-contained
(pure PyTorch + RDKit, no torch_geometric) so it can run on CPU.
"""

from .graph_data import (
    GraphSample,
    build_atom_dataset,
    scaffold_group_split,
)
from .gnn import ForwardGNN13C, pinball_loss

__all__ = [
    "ForwardGNN13C",
    "GraphSample",
    "build_atom_dataset",
    "pinball_loss",
    "scaffold_group_split",
]
