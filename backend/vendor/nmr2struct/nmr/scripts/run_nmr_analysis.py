import argparse
import yaml
import h5py
import numpy as np
import os
from nmr.analysis.analysis_runner import (
    process_substructure_predictions,
    process_SMILES_predictions,
    run_process_parallel
)
from nmr.analysis.util import (
    intake_data,
    apply_substruct_invert_fxn
)
import nmr.analysis.util as analysis_utils
from nmr.analysis.postprocessing import (
    collate_predictions,
    postprocess_save_SMILES_results,
    postprocess_save_substructure_results
)

from .top_level_utils import save_completed_config

def get_args() -> dict:
    parser = argparse.ArgumentParser(description='Run NMR analysis')
    parser.add_argument('config_file', type = str, help = 'The path to the YAML configuration file')
    args = parser.parse_args()
    listdoc = yaml.safe_load(open(args.config_file, 'r'))
    return (
        listdoc['global_args'],
        listdoc['analysis']
    )

def main() -> None:
    print("Parsing arguments...")   
    global_args, analysis_args = get_args()
    file_handles = intake_data(global_args['savedir'], analysis_args['pattern'])
    all_sets = file_handles[0].keys()
    tot_config = {'global_args' : global_args,
                  'analysis_args' : analysis_args}
    save_completed_config("full_analysis_config.yaml", tot_config, global_args['savedir'])

    if analysis_args['analysis_type'] == 'substructure':
        print("Analyzing substructure results")
        result_dict = {}
        with h5py.File(os.path.join(global_args['savedir'], 'combined_predictions.h5'), 'w') as save_handle:
            for set_name in all_sets:
                selected_handles = [f[set_name] for f in file_handles]
                if 'additional_pad_token' in selected_handles[0].keys():
                    addn_pad_token = selected_handles[0]['additional_pad_token'][()]
                else:
                    addn_pad_token = None
                #Gather predictions together because the metrics are computed over all predictions
                #   and targets together
                collated_targets, collated_predictions, collated_smiles = collate_predictions(selected_handles,
                                                                                              pad_tkn = addn_pad_token)
                #For all intents and purposes, substructures should be represented as binary 
                #   arrays for metric calculations. If the rounded predictions are not in the range [0, 1], 
                #   an inversion needs to be done on the sequences to turn them into fixed-length binary arrays.
                unique_rounded_elems = np.unique(collated_predictions.round())
                if unique_rounded_elems.shape[0] != 2 or (not np.allclose(np.unique(collated_predictions.round()), np.array([0, 1]))):
                    print("Inversion required")
                    assert('additional_pad_token' in selected_handles[0].keys())
                    all_pad_tokens = [h['additional_pad_token'][()] for h in selected_handles]
                    assert(len(np.unique(all_pad_tokens)) == 1)
                    pad_token = all_pad_tokens[0]
                    inversion_fxn = getattr(analysis_utils,
                                            analysis_args['substruct_inversion_fxn'])
                    collated_predictions = apply_substruct_invert_fxn(collated_predictions,
                                                                      inversion_fxn=inversion_fxn,
                                                                      padding_token=pad_token)

                try:
                    result_dict[set_name] = process_substructure_predictions(collated_predictions,
                                                                            collated_targets)
                except Exception as e:
                    print(f"Error processing {set_name}: {e}")
                    result_dict[set_name] = "Could not process substructure predictions for this model"
                #Save the collated predictions and targets to the open h5 file pointer
                set_group = save_handle.create_group(set_name)
                set_group.create_dataset('targets', data = collated_targets)
                set_group.create_dataset('predictions', data = collated_predictions)
                set_group.create_dataset('smiles', data = collated_smiles)

            postprocess_save_substructure_results(global_args['savedir'],
                                                result_dict)
    
    elif analysis_args['analysis_type'] == 'SMILES':
        print("Analyzing SMILES results")
        with h5py.File(os.path.join(global_args['savedir'], 'processed_predictions.h5'), 'w') as save_handle:
            for set_name in all_sets:
                selected_handles = [f[set_name] for f in file_handles]
                curr_results = run_process_parallel(process_SMILES_predictions,
                                                    analysis_args['f_addn_args'],
                                                    selected_handles,
                                                    max(16, os.cpu_count()))
                postprocess_save_SMILES_results(save_handle, set_name, curr_results)

if __name__ == "__main__":
    main()