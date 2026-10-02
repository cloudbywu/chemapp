import h5py
import numpy as np
import os
import pickle as pkl
from nmr.analysis.util import count_num_heavy

def collate_predictions(pred_sets: list[h5py.File],
                        pad_tkn: int = None) -> tuple[np.ndarray, np.ndarray]:
    """Combines all the targets and predictions together into one NP array each
    
    In the case that the predictions are not all the same length in the final dimension,
    padding is performed using the given pad_tkn.
    """
    all_targets = []
    all_predictions = []
    all_smiles = []
    for pset in pred_sets:
        assert('targets' in pset.keys())
        assert('predictions' in pset.keys())
        assert('smiles' in pset.keys())
        all_targets.append(pset['targets'])
        all_predictions.append(pset['predictions'])
        all_smiles.append(pset['smiles'])
    #Check prediction length in the last dimension
    pred_lens = [p.shape[-1] for p in all_predictions]
    if len(set(pred_lens)) > 1:
        print("Uneven lengths detected, correcting!")
        max_len = max(pred_lens)
        for i in range(len(all_predictions)):
            curr_len = all_predictions[i].shape[-1]
            if curr_len < max_len:
                pad_amt = max_len - curr_len
                pad_shape = list(all_predictions[i].shape)
                pad_shape[-1] = pad_amt
                pad_arr = np.full(pad_shape, pad_tkn)
                all_predictions[i] = np.concatenate([all_predictions[i], pad_arr], axis = -1)
    return np.concatenate(all_targets), np.concatenate(all_predictions), np.concatenate(all_smiles)

def format_SMILES_preds_into_h5(f: h5py.File, 
                         good_targets: list[str],
                         preds_with_losses: list[list[tuple[str, float]]],
                         bad_targets: list[str],
                         bad_predictions: list[list[str]],
                         bad_scores: list[list[float]]) -> None:
    #Save the bad predictions as a concatenated 2D array to save space
    bad_grp = f.create_group('bad_predictions')
    bad_grp.create_dataset('targets', data = bad_targets)
    bad_grp.create_dataset('predictions', data = np.array(bad_predictions))
    bad_grp.create_dataset('scores', data=np.array(bad_scores))
    #Generate a separate entry for each good prediction
    good_grp = f.create_group('valid_predictions')
    for i in range(len(good_targets)):
        curr_targ = good_targets[i]
        curr_pred = preds_with_losses[i]
        pred_strings = [x[0] for x in curr_pred]
        pred_losses = [x[1] for x in curr_pred]
        pred_scores = [x[2] for x in curr_pred]

        #Saving predictions separately gets around the issue of ragged 2D arrays
        pred_grp = good_grp.create_group(f'{i}')
        pred_grp.create_dataset('target', data = curr_targ)
        pred_grp.create_dataset('prediction_strings', data = pred_strings)
        pred_grp.create_dataset('prediction_bce_losses', data = pred_losses)
        pred_grp.create_dataset('num_heavy_atoms', data = count_num_heavy(curr_targ))
        pred_grp.create_dataset('prediction_scores', data=pred_scores)

def postprocess_save_SMILES_results(f: h5py.File,
                                    savename: str,
                                    processed_results: list[tuple[
                                   list[str], list[list[tuple[str, float]]],
                                   list[str], list[list[str]]]
                                   ]) -> None:
    """Postprocesses the list of results and saves into an h5 file in the savedir for a given set (e.g. train, val, test    )
    
    Args:
        f: open h5py.File object to write data to
        savename: the name to save the predictions under
        processed_results: list of processed results from the process_smiles_predictions() function

    Notes:
        The processed results are a list of tuples containing a number of elements:
            - good_targets: The SMILES strings that were successfully predicted and processed
            - preds_with_losses: The list of list of tuples containing the predictions and their substructure BCE losses
            - bad_targets: The SMILES strings that were unsuccessfully predicted and could not be processed
            - bad_predictions: The SMILES strings that were unsuccessfully predicted and could not be processed
        For optimization, the failures are saved together in one entry as a 2D matrix and the good predictions 
        are saved one entry at a time due potential ragged arrays.
    """
    group = f.create_group(savename)
    good_targets = []
    preds_with_losses = []
    bad_targets = []
    bad_predictions = []
    bad_scores = []
    #Gather everything together
    for result in processed_results:
        good_targets.extend(result[0])
        preds_with_losses.extend(result[1])
        bad_targets.extend(result[2])
        bad_predictions.extend(result[3])
        bad_scores.extend(result[4])
    format_SMILES_preds_into_h5(group, 
                                good_targets, 
                                preds_with_losses, 
                                bad_targets, 
                                bad_predictions,
                                bad_scores)

def postprocess_save_substructure_results(savedir: str,
                                          metrics_dict: dict) -> None:
    """Saves the substructure metrics to the savedir, no processing needed"""
    with open(os.path.join(savedir, 'substructure_metrics.pkl'), 'wb') as f:
        pkl.dump(metrics_dict, f)