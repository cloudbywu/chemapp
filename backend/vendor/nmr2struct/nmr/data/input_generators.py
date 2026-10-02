import numpy as np
from .tokenizer import BasicSmilesTokenizer
from scipy.signal import find_peaks
import warnings

def look_ahead_substructs(labels: np.ndarray) -> int:
    """Determines the maximum sequence length for padding"""
    max_len = 0
    for i in range(len(labels)):
        max_len = max(max_len, np.count_nonzero(labels[i]))
    return max_len

### Spectrum processing methods ###
def threshold_spectra(spectra: np.ndarray, eps: float) -> np.ndarray:
    """Sets values lower than eps to 0"""
    spectra[spectra < eps] = 0
    return spectra

def peaks_with_radius(spectrum: np.ndarray, radius: int) -> np.ndarray:
    """Selects points based on a radius value around peaks.

    Args:
        spectrum: The array of intensities to select peaks from 
        radius: The radius to use for point selection about peaks
    
    This function first uses find_peaks to select out peaks and then includes indices within the radius,
    i.e. i - radius <= x <= i + radius, where i is the peak index and x is the index of the point.
    Because these peaks are expanded to segments that could potentially overlap, we explicitly remove 
    duplicates. 
    """
    peaks, _ = find_peaks(spectrum)
    #+1 to include the upper bound
    all_intervals = [np.arange(i - radius, i + radius + 1) for i in peaks]
    unique_sorted = np.unique(np.concatenate(all_intervals))
    return unique_sorted[(unique_sorted >= 0) & (unique_sorted < len(spectrum))]

def peaks_and_minima(spectrum: np.ndarray) -> np.ndarray:
    """Selects peaks and minima from the spectrum
    The minima are found by flipping the spectral intensities (multiplying by -1) and then
    finding the peaks again. Overlaps are checked, i.e. a peak cannot be a minimum and vice versa.
    """
    peaks, _ = find_peaks(spectrum)
    minima, _ = find_peaks(-spectrum)
    return np.sort(np.concatenate((peaks, minima)))

def spectrum_extraction(spectrum: np.ndarray, criterion: str, radius: int = None) -> np.ndarray:
    """Extracts the indices from a spectrum based on the given criterion
    
    Args:
        spectrum: The array of intensities to select peaks from
        criterion: The method of peak selection
        radius: The radius to use for criterion 'peaks_with_radius'. Defaults to 
            None, and should not be used by any generator besides the threshold tokenized
            or threshold pairs representations.
    """
    if criterion == 'all_nonzero':
        indices = np.where(spectrum > 0)[0]
    elif criterion == 'find_peaks':
        indices, _ = find_peaks(spectrum)
    elif criterion == 'binary':
        indices = np.where(spectrum == 1)[0]
    elif criterion == 'peaks_with_radius':
        assert(radius is not None)
        indices = peaks_with_radius(spectrum, radius)
    elif criterion == 'peaks_and_minima':
        #Select out peaks and the minima that are interspersed between them
        indices = peaks_and_minima(spectrum)
        pass
    else:
        raise ValueError("Invalid criterion for spectrum extraction")
    return indices

def spectrum_ppm_normalization(selected_ppm: np.ndarray,
                               ppm_shift_range: np.ndarray,
                               normalization: str) -> np.ndarray:
    '''Normalizes the ppm shift values of the given spectrum 
    Args:
        selected_ppm: The selected ppm values to map into a standard interval
        ppm_shift_range: The range of true ppm values used in the spectrum
        normalziation: The method for normalizing (cannot be None):
            'neg_uniform': ppm values are normalized to [-1, 0]
            'uniform': ppm values are normalized to [0, 1]
            'first_half': ppm values are normalized to [0, 0.5]
            'second_half': ppm values are normalized to [0.5, 1]
    
    The strategy for normalization is to map the ppm values to the interval [0, 1] and then
    apply the desired transformation to the interval [a, b], i.e. given x:

        x_unif = (x - min(x)) / (max(x) - min(x))
        x_final = x_unif * (b - a) + a
    '''
    min_ppm = np.min(ppm_shift_range)
    max_ppm = np.max(ppm_shift_range)
    ppm_range = max_ppm - min_ppm
    unif_ppms = (selected_ppm - min_ppm) / ppm_range
    if normalization == 'uniform':
        return unif_ppms
    elif normalization == 'neg_uniform':
        a, b = -1, 0
    elif normalization == 'first_half':
        a, b = 0, 0.5
    elif normalization == 'second_half':
        a, b = 0.5, 1
    else:
        raise ValueError("Unsupported normalization scheme")
    return (unif_ppms * (b - a)) + a

def select_points(spectra: np.ndarray, hnmr_criterion: str, hnmr_radius: int, cnmr_criterion: str) -> np.ndarray:
    hnmr_spectrum = spectra[:28000]
    cnmr_spectrum = spectra[28000:28040]
    hnmr_indices = spectrum_extraction(hnmr_spectrum, hnmr_criterion, hnmr_radius)   
    cnmr_indices = spectrum_extraction(cnmr_spectrum, cnmr_criterion, None)
    return hnmr_spectrum, cnmr_spectrum, hnmr_indices, cnmr_indices

def point_representation(representation_name: str,
                         hnmr_spectrum: np.ndarray,
                         cnmr_spectrum: np.ndarray,
                         hnmr_indices: np.ndarray,
                         cnmr_indices: np.ndarray,
                         hnmr_shifts: np.ndarray = None,
                         cnmr_shifts: np.ndarray = None,
                         hnmr_normalization_method: str = None,
                         cnmr_normalization_method: str = None,
                         bins: np.ndarray = None,
                         sep_token: int = None,
                         add_type_feature: bool = False) -> np.ndarray:
    """Transforms the spectrum into the desired representation for downstream tasks
    Args:
        representation_name: The name of the representation to use, currently the following are implemented:
            'tokenized_indices': Returns the toeknized intensity values at the given indices and the indices themselves for 
                specific positional encodings
            'continuous_pair': Returns pairs of (x, y) for the selected points where x is the ppm shift and y is the 
                intensity value
        hnmr_spectrum: Numpy array of the HNMR intensities
        cnmr_spectrum: Numpy array of the CNMR intensities
        hnmr_indices: Numpy array of selected HNMR indices
        cnmr_indices: Numpy array of selected CNMR indices
        hnmr_shifts: Array of HNMR shifts. Required for 'continuous_pair' representation
        cnmr_shifts: Array of CNMR shifts. Required for 'continuous_pair' representation
        hnmr_normalization_method: Method for normalizing the HNMR ppm values
        cnmr_normalization_method: Method for normalizing the CNMR ppm values
        bins: np.ndarray, bin array to use for digitizing spectra
        sep_token: int, a separator token optionally used for separating HNMR from CNMR data in 
            an array. Defaults to None
        add_type_feature: bool, whether to add a type feature to the representation. Defaults to False. 
    
    Notes: A type feature is an additional indicator which specifies where the data point comes from, i.e.
        HNMR or CNMR. This is useful for the transformer model to distinguish between the two types of data.
        In the tokenized case, the type feature is an integer with.
        For tokenized case:
            HNMR: 1
            CNMR: 2
            Separator: 3
        For continuous case: 
            HNMR: 1
            CNMR: 0
    """
    if representation_name == 'tokenized_indices':
        assert(bins is not None)
        if sep_token is None:
            all_indices = np.concatenate((hnmr_indices, cnmr_indices + 28000))
            all_intensities = np.concatenate((hnmr_spectrum[hnmr_indices], cnmr_spectrum[cnmr_indices]))
            tokenized_intensities = np.digitize(all_intensities, bins)
        else:
            #Add the separating token
            assert(sep_token == len(bins) + 1)
            all_indices = np.concatenate((hnmr_indices, [28000], cnmr_indices + 28001))
            hnmr_intensities = np.digitize(hnmr_spectrum[hnmr_indices], bins)
            cnmr_intensities = np.digitize(cnmr_spectrum[cnmr_indices], bins)
            tokenized_intensities = np.concatenate((hnmr_intensities, [sep_token], cnmr_intensities))
        intermediate = np.vstack((tokenized_intensities, all_indices))
        if add_type_feature:
            if sep_token is None:
                hnmr_type_feature = np.ones_like(hnmr_indices)
                cnmr_type_feature = np.ones_like(cnmr_indices) * 2
            else:
                #Add 3 here for the separating token that appears at the end of 
                #   HNMR and before CNMR
                hnmr_type_feature = np.concatenate((np.ones_like(hnmr_indices), np.array([3])))
                cnmr_type_feature = np.ones_like(cnmr_indices) * 2
            type_feature = np.concatenate((hnmr_type_feature, cnmr_type_feature))
            intermediate = np.vstack((intermediate, type_feature))
        return intermediate
    elif representation_name == 'continuous_pair':
        assert((hnmr_shifts is not None) and (cnmr_shifts is not None))
        selected_hnmr_shifts = hnmr_shifts[hnmr_indices]
        if hnmr_normalization_method is not None:
            selected_hnmr_shifts = spectrum_ppm_normalization(selected_hnmr_shifts,
                                                              hnmr_shifts,
                                                              hnmr_normalization_method)
        selected_hnmr_intensities = hnmr_spectrum[hnmr_indices]
        selected_cnmr_shifts = cnmr_shifts[cnmr_indices]
        if cnmr_normalization_method is not None:
            selected_cnmr_shifts = spectrum_ppm_normalization(selected_cnmr_shifts,
                                                              cnmr_shifts,
                                                              cnmr_normalization_method)
        selected_cnmr_intensities = cnmr_spectrum[cnmr_indices]
        hnmr_pairs = np.vstack((selected_hnmr_shifts, selected_hnmr_intensities)).T
        cnmr_pairs = np.vstack((selected_cnmr_shifts, selected_cnmr_intensities)).T
        if add_type_feature:
            hnmr_type_feature = np.ones((len(hnmr_pairs), 1))
            cnmr_type_feature = np.zeros((len(cnmr_pairs), 1))
            hnmr_pairs = np.hstack((hnmr_pairs, hnmr_type_feature))
            cnmr_pairs = np.hstack((cnmr_pairs, cnmr_type_feature))
        return np.vstack((hnmr_pairs, cnmr_pairs))

def apply_padding(representation_name: str, 
                  processed_spectrum: np.ndarray,
                  padding_value: int,
                  max_len: int,
                  padding_variation: str = 'zeros') -> np.ndarray:
    """Applies padding to the processed spectrum using the given padding value
    Args:
        representation_name: The name of the representation to use, consistent with point_representation()
        processed_spectrum: The processed spectrum to pad
        padding_value: The value to use for padding
        max_len: The maximum length to pad to
        padding_variation: How padding is done for the tokenized case. 'zeros' pads with zeros while 
            'complement' pads with the complement of the sequence up to the desired length. Applied to the 
            index portion of the tokenized representation input. Defaults to 'zeros'.
            Note that the 'complement' padding variation assumes a maximum length of 28040.
    """
    if representation_name == 'tokenized_indices':
        if padding_variation == 'zeros':
            return np.pad(
                processed_spectrum,
                ((0, 0), (0, max_len - processed_spectrum.shape[1])),
                'constant',
                constant_values = (padding_value,)
            )
        elif padding_variation == 'complement':
            all_inds = set(np.arange(28040))
            index_values = set(processed_spectrum[1])
            complement = np.sort(list(all_inds - index_values))[:max_len - processed_spectrum.shape[1]]
            padding = np.vstack(
                (np.zeros(len(complement)),
                complement)
            )
            return np.hstack(
                (processed_spectrum, padding)
            )
        else:
            raise ValueError("Unsupported padding variation!")
    elif representation_name == 'continuous_pair':
        return np.vstack((
            processed_spectrum, np.ones((max_len - processed_spectrum.shape[0], processed_spectrum.shape[-1])) * padding_value
        ))

def look_ahead_spectra(spectra: np.ndarray, 
                       hnmr_criterion: str,
                       hnmr_radius: int, 
                       cnmr_criterion: str,
                       eps: float) -> int:
    """Determines the maximum number of peaks for padding"""
    max_hnmr_len = -1
    max_cnmr_len = -1
    max_tot_len = -1
    for i in range(len(spectra)):
        _, _, hnmr_indices, cnmr_indices = select_points(threshold_spectra(spectra[i], eps), 
                                                         hnmr_criterion, 
                                                         hnmr_radius,
                                                         cnmr_criterion)
        max_hnmr_len = max(max_hnmr_len, len(hnmr_indices)) 
        max_cnmr_len = max(max_cnmr_len, len(cnmr_indices))
        tot_len = len(hnmr_indices) + len(cnmr_indices)
        max_tot_len = max(max_tot_len, tot_len)
    #Keep these maximums separate for greater flexibility
    return max_hnmr_len, max_cnmr_len, max_tot_len

#Abstract base class for input generators, to be inherited by others
class InputGeneratorBase:
    #Getters have concrete implementations, but constructor and transform are not implemented
    def __init__(self,
                 spectra: np.ndarray,
                 labels: np.ndarray,
                 smiles: np.ndarray,
                 tokenizer: BasicSmilesTokenizer,
                 alphabet: np.ndarray,
                 eps: float):
        pass

    def transform(self, spectra: np.ndarray, smiles: str, substructures: np.ndarray) -> np.ndarray:
        pass

    def get_size(self) -> int:
        return self.alphabet_size
    
    def get_ctrl_tokens(self) -> tuple[int, int, int]:
        return (self.stop_token, self.start_token, self.pad_token)
    
    def get_max_seq_len(self) -> int:
        return self.max_len

class SubstructureRepresentationOneIndexed(InputGeneratorBase):
    """Processes binary substructure array to 1-indexed values with 0 padding"""
    def __init__(self, 
                 spectra: np.ndarray,
                 labels: np.ndarray,
                 smiles: np.ndarray,
                 tokenizer: BasicSmilesTokenizer,
                 alphabet: np.ndarray,
                 eps: float):
        """
        Args:
            spectra: Numpy array of all spectra
            labels: Numpy array of all substructures
            smiles: Numpy array of all smiles
            tokenizer: Tokenizer to use for smiles
            alphabet: Path to the alphabet file
            eps: Epsilon value for thresholding spectra
        """
        self.pad_token = 0
        self.stop_token = None
        self.start_token = None
        self.max_len = look_ahead_substructs(labels)
        self.alphabet_size = labels.shape[1] + 1
    
    def transform(self, spectra: np.ndarray, smiles: str, substructures: np.ndarray) -> np.ndarray:
        """Transforms the input binary substructure array into shifted and padded 1-indexed array
        Args:
            substructures: Binary substructure array
        
        Example:
            original vector:  [0, 1, 0, 1, 1, 0, 0]
            shifted + padded vector: [2, 4, 5, 0, 0, 0, 0]
        """
        indices = np.arange(len(substructures)) + 1
        indices = indices * substructures
        nonzero_entries = indices[indices != 0]
        padded = np.pad(nonzero_entries, 
                        (0, self.max_len - len(nonzero_entries)), 
                        'constant', 
                        constant_values = (self.pad_token,))
        return padded
    
class SubstructureRepresentationBinary(InputGeneratorBase):
    """Dummy class for binary substructure representation and interface consistency"""
    def __init__(self, 
                 spectra: np.ndarray,
                 labels: np.ndarray,
                 smiles: np.ndarray,
                 tokenizer: BasicSmilesTokenizer,
                 alphabet: np.ndarray,
                 eps: float):
        """
        Args:
            spectra: Numpy array of all spectra
            labels: Numpy array of all substructures
            smiles: Numpy array of all smiles
            tokenizer: Tokenizer to use for smiles
            alphabet: Path to the alphabet file
            eps: Epsilon value for thresholding spectra
        """
        #We set the alphabet to 957 because binary representation requires 957 tokens 
        #   (even though each token is a 0 or 1)
        self.alphabet_size = 957
        self.pad_token = None
        self.stop_token = None
        self.start_token = None
        self.max_len = 957

    def transform(self, spectra: np.ndarray, smiles: str, substructures: np.ndarray) -> np.ndarray:
        """Returns the substructure array"""
        #We expand here because for substructure to structure, the transformer expects a 3D input
        return np.expand_dims(substructures, axis = -1)

class SpectrumRepresentationUnprocessed(InputGeneratorBase):

    def __init__(self, 
                 spectra: np.ndarray,
                 labels: np.ndarray,
                 smiles: np.ndarray,
                 tokenizer: BasicSmilesTokenizer,
                 alphabet: np.ndarray,
                 eps: float):
        """
        Args:
            spectra: Numpy array of all spectra
            labels: Numpy array of all substructures
            smiles: Numpy array of all smiles
            tokenizer: Tokenizer to use for smiles
            alphabet: Path to the alphabet file
            eps: Epsilon value for thresholding spectra
        """
        
        self.pad_token = None
        self.stop_token = None
        self.start_token = None
        self.alphabet_size = 28045
        self.max_len = 28040
    
    def transform(self, spectra: np.ndarray, smiles: str, substructures: np.ndarray) -> np.ndarray:
        """Returns the spectra array"""
        return spectra

class SpectrumRepresentationThresholdTokenized(InputGeneratorBase):
    """Selects peaks from the spectrum after thresholding and tokenizes them"""
    def __init__(self, 
                 spectra: np.ndarray,
                 labels: np.ndarray,
                 smiles: np.ndarray,
                 tokenizer: BasicSmilesTokenizer,
                 alphabet: np.ndarray,
                 eps: float,
                 hnmr_selection: str = 'all_nonzero',
                 hnmr_radius: int = None,
                 cnmr_selection: str = 'all_nonzero',
                 add_hnmr_cnmr_spacing: bool = False,
                 add_type_feature: bool = False,
                 padding_variation: str = 'zeros', 
                 nbins: int = 200):
        """
        Args:
            spectra: Numpy array of all spectra
            labels: Numpy array of all substructures
            smiles: Numpy array of all smiles
            tokenizer: Tokenizer to use for smiles
            alphabet: Path to the alphabet file
            eps: Epsilon value for thresholding spectra
            hnmr_selection: The criterion to use for selecting HNMR peaks. See documentation for 
                spectrum_extraction() for valid arguments
            hnmr_radius: The radius to use for criterion 'peaks_with_radius' when selecting points from the 
                hnmr spectrum around peaks. Defaults to None
            cnmr_selection: The criterion to use for selecting CNMR peaks. See documentation for 
                spectrum_extraction() for valid arguments
            add_hnmr_cnmr_spacing: Whether to add a token between the HNMR and CNMR peaks to 
                indicate a separation between the two. Defaults to False. The separating token is 
                set to 
            add_type_feature: Whether to add an additional token indicating where the spectrum originated from,
                either HNMR or CNMR. Defaults to False
            padding_variation: How padding is done on the sequence. 'zeros' pads with zeros, 
                but 'complement' pads with the complement of the sequence. For instance, suppose a sequence:
                    [1, 3, 5]
                and padding to a maximum length of 5. The complement of the sequence is the set of ordered indices
                [0, max_len - 1], and elements are appended from this complement to achieve the target length:
                    [1,3,5] --> [1, 3, 5, 0, 2]
            nbins: The number of bins to use for digitizing the spectra

        Note: The additional arguments:
            hnmr_selection,
            cnmr_selection,
            nbins
        should be specified in the config file as additional arguments for the input generator

        Spectra are expected as arrays of normalized intensity with values in [0, 1]
        """
        
        self.hnmr_criterion = hnmr_selection
        self.hnmr_radius = hnmr_radius
        self.cnmr_criterion = cnmr_selection
        self.eps = eps
        self.max_hnmr_len, self.max_cnmr_len, self.max_len = look_ahead_spectra(spectra, self.hnmr_criterion, self.hnmr_radius, self.cnmr_criterion, self.eps)
        self.pad_token = 0 
        self.stop_token = None
        self.start_token = None
        self.padding_variation = padding_variation
        self.add_type_feature = add_type_feature
        
        #Account for addition of separating token
        if add_hnmr_cnmr_spacing:
            self.separator_token = nbins + 1
            self.alphabet_size = nbins + 2
            self.max_len += 1 #Add one here to account for separating token
        else:
            self.separator_token = None
            self.alphabet_size = nbins + 1

        self.bins = np.linspace(eps, 1, nbins)
        self.representation_name = 'tokenized_indices'

    def transform(self, spectra: np.ndarray, smiles: str, substructures: np.ndarray) -> np.ndarray:
        spectra = threshold_spectra(spectra, self.eps)
        hnmr_spectrum, cnmr_spectrum, hnmr_indices, cnmr_indices = select_points(spectra, self.hnmr_criterion, self.hnmr_radius, self.cnmr_criterion)
        processed_spectrum = point_representation(self.representation_name,
                                                  hnmr_spectrum,
                                                  cnmr_spectrum,
                                                  hnmr_indices,
                                                  cnmr_indices,
                                                  bins=self.bins,
                                                  sep_token=self.separator_token,
                                                  add_type_feature=self.add_type_feature)
        processed_spectrum = apply_padding(self.representation_name, 
                                           processed_spectrum, 
                                           self.pad_token, 
                                           self.max_len,
                                           padding_variation=self.padding_variation)  
        return processed_spectrum
    
class SpectrumRepresentationThresholdPairs(InputGeneratorBase):
    """Selects ALL non-zero peaks from the spectrum after thresholding and represents them as pairs"""
    def __init__(self, spectra: np.ndarray,
                 labels: np.ndarray,
                 smiles: np.ndarray,
                 tokenizer: BasicSmilesTokenizer,
                 alphabet: np.ndarray,
                 eps: float,
                 hnmr_selection: str = 'all_nonzero',
                 hnmr_radius: int = None,
                 cnmr_selection: str = 'all_nonzero',
                 hnmr_shifts: str = None,
                 cnmr_shifts: str = None,
                 hnmr_normalization: str = None,
                 cnmr_normalization: str = None,
                 add_type_feature: bool = False):
        """
        Args:
            spectra: Numpy array of all spectra
            labels: Numpy array of all substructures
            smiles: Numpy array of all smiles
            tokenizer: Tokenizer to use for smiles
            alphabet: Path to the alphabet file
            eps: Epsilon value for thresholding spectra
            hnmr_selection: The criterion to use for selecting HNMR peaks. See documentation for 
                spectrum_extraction() for valid arguments
            hnmr_radius: The radius to use for criterion 'peaks_with_radius' when selecting points from the 
                hnmr spectrum around peaks. Defaults to None
            cnmr_selection: The criterion to use for selecting CNMR peaks. See documentation for 
                spectrum_extraction() for valid arguments
            hnmr_shifts: Path to a file specifying the HNMR shift values
            cnmr_shifts: Path to a file specifying the CNMR shift values
            hnmr_normalization: Method for normalizing the HNMR ppm values:
                None: no normalization is performed
                'uniform': ppm values are normalized to [0, 1]
                'neg_uniform': ppm values are normalized to [-1, 0]
                'first_half': ppm values are normalized to [0, 0.5]
                'second_half': ppm values are normalized to [0.5, 1]
            cnmr_normalization: Method for normalizing CNMR shift values. Same 
                options as hnmr_normalization.
            add_type_feature: Whether to add an additional value indicating where the spectrum originated from,
                either HNMR or CNMR. Defaults to False

        Note: The additional arguments:
            hnmr_selection,
            cnmr_selection,
            hnmr_shifts,
            cnmr_shifts
        should be specified in the config file as additional arguments for the input generator

        Spectra are expected as arrays of normalized intensity with values in [0, 1]
        """
        #Handle shift initialization
        if hnmr_shifts is not None:
            self.hnmr_shifts = np.load(hnmr_shifts, allow_pickle=True)
        else:
            warnings.warn("No HNMR shifts provided, using default values from -2 to 12 ppm")
            self.hnmr_shifts = np.arange(-2, 12, 0.0005)
        if cnmr_shifts is not None:
            self.cnmr_shifts = np.load(cnmr_shifts, allow_pickle=True)
        else:
            warnings.warn("No CNMR shifts provided, using default values from -2 to 231 ppm")
            self.cnmr_shifts = np.linspace(-2, 231, 40)
        
        self.hnmr_criterion = hnmr_selection
        self.cnmr_criterion = cnmr_selection
        self.hnmr_normalization = hnmr_normalization
        self.hnmr_radius = hnmr_radius
        self.cnmr_normalization = cnmr_normalization
        self.eps = eps
        self.max_hnmr_len, self.max_cnmr_len, self.max_len = look_ahead_spectra(spectra, self.hnmr_criterion, self.hnmr_radius, self.cnmr_criterion, self.eps)
        self.pad_token = -1000 
        self.stop_token = None
        self.start_token = None
        #Not relevant for this representation
        self.alphabet_size = self.max_len
        self.representation_name = 'continuous_pair'
        self.add_type_feature = add_type_feature

    def transform(self, spectra: np.ndarray, smiles: str, substructures: np.ndarray) -> np.ndarray:
        spectra = threshold_spectra(spectra, self.eps)
        hnmr_spectrum, cnmr_spectrum, hnmr_indices, cnmr_indices = select_points(spectra, self.hnmr_criterion, self.hnmr_radius, self.cnmr_criterion)
        processed_spectrum = point_representation(self.representation_name,
                                                  hnmr_spectrum,
                                                  cnmr_spectrum,
                                                  hnmr_indices,
                                                  cnmr_indices,
                                                  hnmr_shifts=self.hnmr_shifts,
                                                  cnmr_shifts=self.cnmr_shifts,
                                                  hnmr_normalization_method=self.hnmr_normalization,
                                                  cnmr_normalization_method=self.cnmr_normalization,
                                                  add_type_feature=self.add_type_feature)
        processed_spectrum = apply_padding(self.representation_name,
                                           processed_spectrum,
                                           self.pad_token,
                                           self.max_len)
        return processed_spectrum
    