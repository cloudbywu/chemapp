import torch
from torch.utils.data import DataLoader
import argparse
import yaml
from nmr.data import create_dataset
from nmr.models import create_model
from nmr.inference.inference import run_inference
from .top_level_utils import (
    seed_everything,
    seed_worker,
    dtype_convert,
    split_data_subsets,
    save_inference_predictions,
    save_token_size_dict,
    specific_update,
    select_model,
    save_completed_config,
    divide_parallel_subsets
)
from typing import Any

#Necessary functions for distributed data parallel inference

def get_args() -> dict:
    '''Parses the passed yaml file to get arguments'''
    parser = argparse.ArgumentParser(description='Run NMR inference')
    parser.add_argument('config_file', type = str, help = 'The path to the YAML configuration file')
    parser.add_argument('local_rank', type = int, help = 'The local rank of this process')
    parser.add_argument('n_procs', type = int, help = 'The total number of concurrent processes running')
    args = parser.parse_args()
    listdoc =  yaml.safe_load(open(args.config_file, 'r'))
    return (
        listdoc['global_args'],
        listdoc['data'],
        listdoc['model'],
        listdoc['inference']
    ), (args.local_rank, args.n_procs)

def main() -> None:
    print("Parsing arguments...")
    args, ids = get_args()
    global_args, dataset_args, model_args, inference_args = args
    local_rank, n_procs = ids

    seed = seed_everything(global_args['seed'])

    dtype = dtype_convert(global_args['dtype'])
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    #Updated logic to handle multiple datasets (the files paths are now lists)
    datasets, updated_dataset_args = create_dataset(dataset_args, dtype, device)
    #Only one dataset needs to save the alphabet
    datasets[0].save_smiles_alphabet(global_args['savedir'])
    #Check and collapse size dictionaries (affects model architecture)
    size_dicts = [d.get_sizes() for d in datasets]
    assert all([s == size_dicts[0] for s in size_dicts])
    size_dict = size_dicts[0]
    #Check and collapse token dictionaries (affects training)
    token_dicts = [d.get_ctrl_tokens() for d in datasets]
    assert all([t == token_dicts[0] for t in token_dicts])
    token_dict = token_dicts[0]
    #Max length dictionaries do not need adjusting
    max_len_dicts = [d.get_max_seq_len() for d in datasets] 
    total_dict = {**size_dict, **token_dict, 'max_lengths' : max_len_dicts}

    inference_args = specific_update(inference_args, total_dict)
    model_args = specific_update(model_args, total_dict)
    total_dict['seed'] = seed

    model, updated_model_args = create_model(model_args, dtype, device)
    model.to(dtype).to(device)

    #Find and load the best model checkpoint
    mod_ckpt_name = select_model(global_args['savedir'],
                                 inference_args['model_selection'])
    print(f"Using the model checkpoint {mod_ckpt_name}")
    best_model_ckpt = torch.load(mod_ckpt_name, map_location = device)
    model.load_state_dict(best_model_ckpt['model_state_dict'])

    if local_rank == 0:
        #Only do this under the first process
        tot_config = {
            'global_args' : global_args,
            'data' : updated_dataset_args,
            'model' : updated_model_args,
            'inference' : inference_args
        }
        save_completed_config('full_inference_config.yaml', tot_config, global_args['savedir'])
        save_token_size_dict(global_args['savedir'], total_dict, 'inference')

    #Set up dataloaders
    splits = inference_args['splits']
    assert len(splits) == len(datasets)
    train_sets, val_sets, test_sets = [], [], []
    for i, s in enumerate(splits):
        train_set, val_set, test_set = split_data_subsets(datasets[i],
                                                          s, 
                                                          inference_args['train_size'],
                                                          inference_args['val_size'],
                                                          inference_args['test_size'])
        train_sets.append(train_set)
        val_sets.append(val_set)
        test_sets.append(test_set)
        
    #Further divide the data here, doing this process for each dataset
    for i in range(len(train_sets)):
        curr_train_set = train_sets[i]
        curr_val_set = val_sets[i]
        curr_test_set = test_sets[i]

        curr_train_set = divide_parallel_subsets(curr_train_set, n_procs, local_rank)
        curr_val_set = divide_parallel_subsets(curr_val_set, n_procs, local_rank)
        curr_test_set = divide_parallel_subsets(curr_test_set, n_procs, local_rank)
    
        g = torch.Generator()
        g.manual_seed(0)

        if len(curr_train_set) > 0:
            curr_train_loader = DataLoader(curr_train_set, 
                                    worker_init_fn=seed_worker,
                                    generator=g,
                                    **inference_args['dloader_args'])
        else:
            curr_train_loader = []

        if len(curr_val_set) > 0:
            curr_val_loader = DataLoader(curr_val_set, 
                                    worker_init_fn=seed_worker,
                                    generator=g,
                                    **inference_args['dloader_args'])
        else:
            curr_val_loader = []

        if len(curr_test_set) > 0:
            curr_test_loader = DataLoader(curr_test_set, 
                                    worker_init_fn=seed_worker,
                                    generator=g,
                                    **inference_args['dloader_args'])
        else:
            curr_test_loader = []

        #Run inference
        print(f"Running inference for dataset {i}...")
        if ('train' in inference_args['sets_to_run']):
            train_predictions = run_inference(model,
                                            curr_train_loader,
                                            device=device, 
                                            **inference_args['run_inference_args']
                                            )
        else:
            train_predictions = None
        
        if ('val' in inference_args['sets_to_run']):
            val_predictions = run_inference(model,
                                            curr_val_loader,
                                            device=device, 
                                            **inference_args['run_inference_args']
                                            )
        else:
            val_predictions = None
            
        if ('test' in inference_args['sets_to_run']):
            test_predictions = run_inference(model,
                                            curr_test_loader,
                                            device=device, 
                                            **inference_args['run_inference_args']
                                            )
        else:
            test_predictions = None

        #Wait for everyone to get to the same point before saving predictions
            
        save_inference_predictions(global_args['savedir'], 
                                train_predictions,
                                val_predictions,
                                test_predictions,
                                idx = local_rank,
                                savetag=f"dataset_{i}")

if __name__ == '__main__':
    main()