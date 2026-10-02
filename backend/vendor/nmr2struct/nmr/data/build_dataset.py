from nmr.data.dataset_base import NMRDataset
from typing import List
import torch

def create_dataset(dataset_args: dict, dtype: torch.dtype, device: torch.device) -> List[NMRDataset]:
    """
    Create potentially multiple datasets based on the arguments passed. Here, dataset_args
    should be a dictionary that contains the following keys:

    spectra_file: List of paths to HDF5 files with spectra
    label_file: List of paths to HDF5 files with substructure labels
    smiles_file: List of npy files with the SMILES strings

    input_generator: Function for computing the model input
    input_generator_addn_args: Additional arguments for the input generator
    target_generator: Function for computing the model target
    target_generator_addn_args: Additional arguments for the target generator
    alphabet: Path to the alphabet file
    eps: Epsilon value for thresholding spectra

    There should be a 1-to-1 correspondence between the elements in the list. Also, all 
    data is expected to use (for now) the same input processing, target processing, and all
    smiles should be representable using the same alphabet.
    """
    assert "spectra_file" in dataset_args
    assert "label_file" in dataset_args
    assert "smiles_file" in dataset_args

    assert len(dataset_args["spectra_file"]) == len(dataset_args["label_file"]) == len(dataset_args["smiles_file"])
    num_datasets = len(dataset_args['spectra_file'])
    datasets = []
    for i in range(num_datasets):
        current_spectra = dataset_args['spectra_file'][i]
        current_smiles = dataset_args['smiles_file'][i]
        current_labels = dataset_args['label_file'][i]
        remaining_args = {
            k : dataset_args[k] for k in dataset_args if k not in ['spectra_file', 'smiles_file', 'label_file']
        }
        datasets.append(
            NMRDataset(dtype = dtype,
                       device = device,
                       spectra_file = current_spectra,
                       smiles_file = current_smiles,
                       label_file = current_labels,
                       **remaining_args)
        )
    return datasets, dataset_args