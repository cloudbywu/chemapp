import h5py
import numpy as np
import torch
from rdkit import Chem as chem
from torch.utils.data import Dataset
from sklearn.utils import class_weight


class SpectraHDF5Dataset(Dataset):
    """Spectra dataset."""
    def __init__(self, spectra_file, smiles_file, label_file):
        """
        Args:
            spectra_file (string): Path to the HDF5 file with spectra.
            smiles_file (string): Path to the HDF5 file with smiles.
            label_file (string): Path to the HDF5 file with substructure labels.
        """
        self.spectra_h5 = h5py.File(spectra_file, "r")
        self.label_h5 = h5py.File(label_file, "r")
        
        # Extract the corresponding h5 datasets
        self.spectra = self.spectra_h5["spectra"]
        self.smiles = np.load(smiles_file, allow_pickle=True)
        self.labels = self.label_h5["substructure_labels"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        spectra_data = torch.from_numpy(self.spectra[idx]).float().unsqueeze(0)
        smiles_data = chem.CanonSmiles(self.smiles[idx].decode("utf-8"))
        label_data = torch.from_numpy(self.labels[idx]).float()
        return (spectra_data, smiles_data), (label_data,)
   
class SparseSpectraHDF5Dataset(Dataset):
    """Spectra dataset with sparse encoding."""
    def __init__(self, sparse_spectra_file, smiles_file, label_file):
        """
        Args:
            spectra_file (string): Path to the HDF5 file with spectra.
            smiles_file (string): Path to the HDF5 file with smiles.
            label_file (string): Path to the HDF5 file with substructure labels.
        """
        self.spectra_h5 = h5py.File(sparse_spectra_file, "r")
        self.smiles_h5 = h5py.File(smiles_file, "r")
        self.label_h5 = h5py.File(label_file, "r")
        
        # 
        self.spectra = self.spectra_h5["spectra"]
        self.smiles = self.smiles_h5["strings"]
        self.labels = self.label_h5["substructure_labels"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        raise NotImplementedError()
