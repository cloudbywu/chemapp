import torch
import numpy as np
from torch import nn, Tensor
from typing import Tuple, Callable

class H1Embed(nn.Module):
    """
    1D convolutional neural network for processing the 1HNMR spectrum, consistent with the 
    architecture in the paper "A framework for automated structure elucidation from routine NMR spectra"
    """
    def __init__(self):
        super().__init__()
        #Kernel size = 5, Filters (out channels) = 64, in channels = 1
        self.conv1 = nn.Conv1d(1, 64, 5, stride = 1, padding = 'valid')
        #Max pool of size 12 with stride 12
        self.pool1 = nn.MaxPool1d(12)
        #Kernel size of 9, Filters (out channels) = 128, in channels = 64
        self.conv2 = nn.Conv1d(64, 128, 9, stride = 1, padding = 'valid')
        #Max pool of size 20 with stride 20
        self.pool2 = nn.MaxPool1d(20)
        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten(start_dim = 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        #Linear layers
        self.linear1 = nn.Linear(14848, 256)

        self.pretranspose = nn.Sequential(self.conv1, self.relu, self.pool1, 
                                          self.conv2, self.relu, self.pool2)
        
        self.posttranspose = nn.Sequential(self.flatten, self.dropout, self.linear1, self.relu)
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.pretranspose(x)
        x = torch.transpose(x, 1, 2)
        return self.posttranspose(x)
        
class C13Embed(nn.Module):
    """ Linear + ReLU module for taking 13CNMR information """
    def __init__(self, n_Cfeatures: int):
        super().__init__()
        self.linear = nn.Linear(n_Cfeatures, 36)
    
    def forward(self, x: Tensor) -> Tensor:
        return nn.ReLU()(self.linear(x).squeeze(1))
    
class MolEmbed(nn.Module):
    """ Linear + ReLU module for taking chemical formula information """
    def __init__(self, n_molfeatures: int):
        super().__init__()
        self.linear = nn.Linear(n_molfeatures, 8)
    
    def forward(self, x: Tensor) -> Tensor:
        return nn.ReLU()(self.linear(x).squeeze(1))
    
class MultiHeadOutput(nn.Module):
    """
    Passes theifinal output through a series of linear layers to generate 
    probabiltiies for each substructure. The output is automatically 
    concatenated into a matrix of shape (batch_size, n_substructures)

    Args:  
        n_substructures: The number of substructures to predict for.
    """
    def __init__(self, n_substructures: int):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Linear(1024, 1) for _ in range(n_substructures)
        ])
    
    def forward(self, x: Tensor) -> Tensor:
        outputs = [nn.Sigmoid()(head(x)) for head in self.heads]
        return torch.cat(outputs, dim = -1)

class NMRConvNet(nn.Module):

    """
    A translation of the CNN model used for interpreting spectral and chemical inputs in the first paper. 
    When rewriting the model from Keras to PyTorch, be sure to transpose the data before flattening it since 
    there is a mismatch of variables between the Keras and PyTorch specifications of the 1D convoluational layers 
    (see https://discuss.pytorch.org/t/how-to-transform-conv1d-from-keras/13738/6)
    """
    model_id = 'CNN'
    
    def __init__(self, n_spectral_features: int, n_Cfeatures: int, n_molfeatures: int, n_substructures: int,
                 dtype: torch.dtype = torch.float, device: torch.device = None):
        """
        Args:
            n_spectral_features: The number of spectral features, i.e. 28000
            n_Cfeatures: The number of CNMR features, i.e. 40
            n_molfeatures: The number of chemical formula features, i.e. 5
            n_substructures: The number of substructures to predict for. This is used for 
                constructing a single linear head for each substructure
            dtype: Model datatype. Default is torch.float
            device: Model device. Default is None
        """
        super().__init__()
        self.n_Cfeatures = n_Cfeatures
        self.n_molfeatures = n_molfeatures
        self.n_spectral_features = n_spectral_features
        self.dtype = dtype
        self.device = device

        self.h1_embed = H1Embed()
        self.relu = nn.ReLU()

        tot_num = 256
        if self.n_Cfeatures > 0:
            self.c13_embed = C13Embed(self.n_Cfeatures)
            tot_num += 36
        
        if self.n_molfeatures > 0:
            self.mol_embed = MolEmbed(self.n_molfeatures)
            tot_num += 8
        
        self.linear2 = nn.Linear(tot_num, 1024)
        self.linear3 = nn.Linear(1024, 1024)

        self.out = MultiHeadOutput(n_substructures)

    def _sanitize_forward_args(self, x: tuple[Tensor, tuple[str]]) -> Tensor:
        """Prepares input for use in forward()
        Args:
            x: The input to the model
        """
        #Unpack the tuple
        x, _ = x
        if len(x.shape) == 2:
            x = torch.unsqueeze(x, 1)
        spectral_x = x[:, :, :self.n_spectral_features]
        cnmr_x = x[:, :, self.n_spectral_features:self.n_spectral_features + self.n_Cfeatures]
        mol_x = x[:, :, self.n_spectral_features + self.n_Cfeatures:]
        return spectral_x, cnmr_x, mol_x

    def forward(self, x: tuple[Tensor, tuple[str]]) -> Tensor:
        """
        Args:
            x: ((batch_size, 1, seq_len), smiles)
        """
        spectral_x, cnmr_x, mol_x = self._sanitize_forward_args(x)

        spectral_x = self.h1_embed(spectral_x)

        # Mix in the information from the CNMR and chemical formula
        if self.n_Cfeatures > 0:
            cnmr_x = self.c13_embed(cnmr_x)
            spectral_x = torch.cat((spectral_x, cnmr_x), dim = -1)

        # Preserve the option to include features from the chemical formula
        if self.n_molfeatures > 0:
            mol_x = self.mol_embed(mol_x)
            spectral_x = torch.cat((spectral_x, mol_x), dim = -1)

        spectral_x = self.linear2(spectral_x)
        spectral_x = self.relu(spectral_x)
        spectral_x = self.linear3(spectral_x)
        spectral_x = self.relu(spectral_x)

        # TODO: test if this is necessary
        spectral_x = self.out(spectral_x)

        return spectral_x
    
    def get_loss(self, 
                 x: Tuple[Tensor, Tuple], 
                 y: Tuple[Tensor], 
                 loss_fn: Callable[[Tensor, Tensor], Tensor]) -> Tensor:
        """
        Unpacks the input and obtains the loss value
        Args:
            x: Tuple of a tensor (input) and the set of smiles strings (smiles)
            y: A tuple of a single tensor contaning the target values (labels)
            loss_fn: The loss function to use for the model, with the signature
                tensor, tensor -> tensor
        """
        y_target, = y
        pred = self.forward(x)
        return loss_fn(pred, y_target.to(self.dtype).to(self.device))
