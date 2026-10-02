import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

class SubsWeightedBCELoss(nn.Module):
    def __init__(self,
                 weights: Optional[Tensor] = None):
        super().__init__()
        self.weights = weights
    
    def forward(self, 
                y_pred: Tensor,
                y_true: Tensor) -> Tensor:
        '''
        Computes the weighted BCE Loss with a different 0/1 class weight per substructure. 

        y_pred: (batch_size, num_substructures), the predicted probabilities for each substructure
        y_true: (batch_size, num_substructures), the true substructures present in each molecule
        weights: (batch_size, num_substructures, 2), the weights for each substructure. Each row i represents the 
            0 and 1 weight for substructure i. If weights are not given, then the loss is unweighted.
        reduction: The method for reducing the loss. Defaults to 'mean', but can be 'mean' or 'sum'.
        '''
        criterion = nn.BCELoss(reduction = 'none')
        unweighted_loss = criterion(y_pred, y_true)
        #Compute the weight for each substructure now
        if self.weights is not None:
            w = y_true * self.weights[:,:, 1] + (1 - y_true) * self.weights[:,:, 0]
            weighted_loss = w * unweighted_loss
        else:
            weighted_loss = 1 * unweighted_loss
        #Averaged across substructures
        rowwise_meaned = torch.mean(weighted_loss, dim = 0)
        assert(rowwise_meaned.shape == (957,))
        #Sum for total loss
        tot_loss = torch.sum(rowwise_meaned)
        return tot_loss

CrossEntropyLoss = nn.CrossEntropyLoss
BCELoss = nn.BCELoss

class MultiTaskLoss(nn.Module):
    def __init__(self, 
                 ignore_index: int,
                 substructure_weight: float,
                 structure_weight: float):
        super().__init__()
        self.ignore_index = ignore_index
        self.SUBSTRUCTURE_weight = substructure_weight  
        self.STRUCT_weight = structure_weight
        self.structure_loss = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self.substructure_loss = nn.BCELoss()
    
    def forward(self, 
                mode: str,
                y_pred: Tensor,
                y_true: Tensor) -> Tensor:
        if mode == 'substructure':
            return self.SUBSTRUCTURE_weight * self.substructure_loss(y_pred, y_true)
        elif mode == 'structure':
            return self.STRUCT_weight * self.structure_loss(y_pred, y_true)