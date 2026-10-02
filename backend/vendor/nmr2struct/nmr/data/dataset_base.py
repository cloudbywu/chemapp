from torch.utils.data import Dataset
import numpy as np
import h5py
from typing import List, Tuple
from nmr.data.tokenizer import BasicSmilesTokenizer
from functools import reduce
from rdkit import Chem as chem
import nmr.data.input_generators as input_generators
import nmr.data.target_generators as target_generators
import torch
from tqdm import tqdm
import pickle

class NMRDataset(Dataset):
    """Sketch of potential dataset class"""
    def __init__(self, 
                 spectra_file: str,
                 smiles_file: str,
                 label_file: str, 
                 input_generator: str,
                 input_generator_addn_args: dict,
                 target_generator: str, 
                 target_generator_addn_args: dict,
                 alphabet: str,
                 eps: float = 0.005,
                 front_load_data_processing: bool = False,
                 dtype: torch.dtype = torch.float,
                 device: torch.device = None):
        """
        Args:
            spectra_file: Path to the HDF5 file with spectra
            smiles_file: Path to the HDF5 file with smiles
            label_file: Path to the HDF5 file with substructure labels
            input_generator: Function name that generates the model input
            target_generator: Function name that generates the model target
            alphabet: Path to the alphabet file
            eps: Epsilon value for thresholding spectra
            front_load_data_processing: Flag that indicates whether all data is processed
                at the start (and __getitem__ just returns the processed data) or whether
                items are processed one at a time through __getitem__. The former is faster
                but requires more memory.
        """
        #Very general approach for handling data, but some of the three data sources should
        #   be present in general.
        #Spectra
        if spectra_file is not None:
            self.spectra_h5 = h5py.File(spectra_file, 'r')
            self.spectra = self.spectra_h5['spectra']
        else:
            self.spectra = None
        #Substructures
        if label_file is not None:
            self.label_h5 = h5py.File(label_file, 'r')
            self.labels = self.label_h5['substructure_labels']
        else:
            self.labels = None
        #Smiles
        assert smiles_file is not None
        self.smiles = np.load(smiles_file, allow_pickle = True)

        #Canonicalize up front
        self.smiles = [chem.CanonSmiles(smi.decode('utf-8')) for smi in self.smiles]
        self.tokenizer = BasicSmilesTokenizer()
        self.eps = eps
        self.front_load_data_processing = front_load_data_processing

        self.device = device
        self.dtype = dtype

        if alphabet is not None:
            self.alphabet = np.load(alphabet, allow_pickle = True)
        else:
            self.alphabet = self._determine_smiles_alphabet(self.smiles)

        #Initialize the input and target generators with a quick "look ahead" to 
        #   determine global information, i.e. max lengths and padding
        self.input_generator = getattr(input_generators, input_generator)(
            self.spectra, self.labels, self.smiles, 
            self.tokenizer, self.alphabet, self.eps,
            **input_generator_addn_args
        )
        self.target_generator = getattr(target_generators, target_generator)(
            self.spectra, self.labels, self.smiles, 
            self.tokenizer, self.alphabet, self.eps,
            **target_generator_addn_args
        )

        if self.front_load_data_processing:
            print("Preprocessing all data at once...")
            self.preprocessed_model_inputs = []
            self.preprocessed_model_targets = []
            for i in tqdm(range(len(self))):
                #Empty array to represent null
                spectra_data = self.spectra[i] if self.spectra is not None else np.array([])
                label_data = self.labels[i] if self.labels is not None else np.array([])
                smiles_data = self.smiles[i] 

                model_input = self.input_generator.transform(spectra_data, smiles_data, label_data)
                model_target = self.target_generator.transform(spectra_data, smiles_data, label_data)
                model_input = torch.from_numpy(model_input).to(self.dtype)
                model_target = tuple([torch.from_numpy(elem).to(self.dtype) for elem in model_target])
                self.preprocessed_model_inputs.append(model_input)
                self.preprocessed_model_targets.append(model_target)

    def __len__(self):
        return len(self.smiles)

    def _determine_smiles_alphabet(self, smiles: list[str]):
        """Generates the alphabet from the set of smiles strings
        Args:
            smiles: list of smiles strings
        """
        token_sets = [set(self.tokenizer.tokenize(smi)) for smi in smiles]
        final_set = list(reduce(lambda x, y : x.union(y), token_sets))
        alphabet = sorted(final_set)
        return alphabet
    
    def save_smiles_alphabet(self, save_dir: str) -> None:
        '''Quickly saves the alphabet as a npy file'''
        with open(f"{save_dir}/alphabet.npy", "wb") as f:
            np.save(f, self.alphabet)
    
    def __getitem__(self, idx):
        if self.front_load_data_processing:
            model_input = self.preprocessed_model_inputs[idx].to(self.device)
            model_target = tuple([elem.to(self.device) for elem in self.preprocessed_model_targets[idx]])
            return (model_input, self.smiles[idx]), model_target
        else:
            spectra_data = self.spectra[idx]
            smiles_data = self.smiles[idx]
            label_data = self.labels[idx]
            #Note: model_input is a Tensor, model_target is a tuple of Tensors!
            model_input = self.input_generator.transform(spectra_data, smiles_data, label_data)
            model_target = self.target_generator.transform(spectra_data, smiles_data, label_data)
            model_input = torch.from_numpy(model_input).to(self.dtype).to(self.device)
            model_target = tuple([torch.from_numpy(elem).to(self.dtype).to(self.device) for elem in model_target])
            return (model_input, smiles_data), model_target
    
    def get_sizes(self) -> dict[int, int]:
        """Returns the padding tokens for the input and target"""
        input_size = self.input_generator.get_size()
        target_size = self.target_generator.get_size()
        return {'source_size' : input_size, 'target_size' : target_size}
    
    def get_ctrl_tokens(self) -> dict[str, int]:
        """Returns the stop, start, and pad tokens for the input and target as dicts"""
        input_tokens = self.input_generator.get_ctrl_tokens()
        target_tokens = self.target_generator.get_ctrl_tokens()
        src_stop_token, src_start_token, src_pad_token = input_tokens
        tgt_stop_token, tgt_start_token, tgt_pad_token = target_tokens
        token_dict = {
            'src_stop_token'  : src_stop_token,
            'src_start_token' : src_start_token,
            'src_pad_token'   : src_pad_token,
            'tgt_stop_token'  : tgt_stop_token,
            'tgt_start_token' : tgt_start_token,
            'tgt_pad_token'   : tgt_pad_token
        }
        return token_dict
    
    def get_max_seq_len(self) -> dict[str, int]:
        """Returns the max sequence length for the input and target as dicts"""
        input_len = self.input_generator.get_max_seq_len()
        target_len = self.target_generator.get_max_seq_len()
        return {'max_src_len' : input_len, 'max_tgt_len' : target_len}
    
class SingleNMRDataset(Dataset):
    """Very simple dataset used for processing a single example at a time"""
    def __init__(self, 
                 hnmr_file: str,
                 cnmr_file: str,
                 normalize: bool = True,
                 hnmr_shifts: str = None,
                 cnmr_shifts: str = None,
                 dtype: torch.dtype = torch.float,
                 device: torch.device = None):
        """
        Args:
            hnmr_file: Path to the TXT file with HNMR data
            cnmr_file: Path to the TXT file with CNMR data
            normalize: If True, normalize the HNMR intensities by the highest peak
        """
        assert not (hnmr_file is None and cnmr_file is None)
        assert (hnmr_shifts is not None) and (cnmr_shifts is not None)
        self.hnmr_file = hnmr_file
        self.cnmr_file = cnmr_file
        with open(cnmr_shifts, 'rb') as f:
            self.cnmr_shifts = pickle.load(f)
        with open(hnmr_shifts, 'rb') as f:
            self.hnmr_shifts = pickle.load(f)
        self.normalize = normalize
        self.dtype = dtype
        self.device = device
        self.spectra = self.process_spectra()

    def extract_hnmr_data(self, hnmr_txt_file: str) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts the HNMR data from the file"""
        with open(hnmr_txt_file, 'r') as f:
            lines = f.readlines()
        shift, intensity = [], []
        for line in lines:
            values = line.split()
            ppm, intens = values
            shift.append(float(ppm))
            intensity.append(float(intens))
        return np.array(shift), np.array(intensity)
    
    def process_hnmr(self, hnmr_txt_file: str):
        h_ppm, h_intens = self.extract_hnmr_data(hnmr_txt_file)
        #Reverse both ppm and intensity if they are not ascending order
        ppm_diffs = np.diff(h_ppm)
        if np.any(ppm_diffs < 0):
            h_ppm = h_ppm[::-1]
            h_intens = h_intens[::-1]
        new_h_intens = np.interp(self.hnmr_shifts, h_ppm, h_intens)
        if self.normalize:
            new_h_intens = new_h_intens / np.max(new_h_intens)
        return new_h_intens
    
    def process_cnmr(self, cnmr_txt_file: str):
        with open(cnmr_txt_file, 'r') as f:
            content = f.read()
        #ASSUME for now that CNMR is a comma separated list of ppm shifts
        all_vals = np.array([float(val) for val in content.split(',')])
        all_vals = np.sort(all_vals)
        bins = np.digitize(all_vals, self.cnmr_shifts)
        bins = np.where(bins == len(self.cnmr_shifts), len(self.cnmr_shifts) - 1, bins)
        spectrum = np.zeros(len(self.cnmr_shifts))
        spectrum[bins] = 1
        return spectrum
    
    def process_spectra(self):
        if self.hnmr_file is not None:
            hnmr_spectra = self.process_hnmr(self.hnmr_file)
        else:
            hnmr_spectra = np.zeros(len(self.hnmr_shifts))
        if self.cnmr_file is not None:
            cnmr_spectra = self.process_cnmr(self.cnmr_file)
        else:
            cnmr_spectra = np.zeros(len(self.cnmr_shifts))
        return np.concatenate([hnmr_spectra, cnmr_spectra])
    
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        return (torch.from_numpy(self.spectra).to(self.dtype).to(self.device), 
                "NULL"), (torch.tensor(0).to(self.dtype).to(self.device), 
                         torch.tensor(0).to(self.dtype).to(self.device))