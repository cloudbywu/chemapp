import torch
from torch.utils.data import DataLoader
import argparse
import yaml
from nmr.data import SingleNMRDataset
from nmr.models import create_model
from nmr.inference.inference import run_inference
from .top_level_utils import (
    seed_everything,
    dtype_convert,
    save_inference_predictions,
)
from typing import Any
import os

def none_or_str(value):
    if value.lower() == 'none':
        return None
    return value

def get_args() -> dict:
    '''Parses the passed yaml file to get arguments'''
    parser = argparse.ArgumentParser(description='Run NMR inference')
    parser.add_argument('--config', type = str, help = 'The path to the YAML configuration file that contains model and inference parameters')
    parser.add_argument('--hnmr_file', type = none_or_str, help = 'The path to the HNMR file')
    parser.add_argument('--cnmr_file', type = none_or_str, help = 'The path to the CNMR file')
    parser.add_argument('--hnmr_shifts', type = str, help = 'The path to the HNMR shifts')
    parser.add_argument('--cnmr_shifts', type = str, help = 'The path to the CNMR shifts')
    parser.add_argument('--ckpt', type = str, help = 'The path to the model checkpoint to use for inference')
    parser.add_argument('--normalize', action='store_true', help='Whether to normalize the HNMR spectra')
    args = parser.parse_args()
    listdoc =  yaml.safe_load(open(args.config, 'r'))
    return (
        listdoc['global_args'],
        listdoc['model'],
        listdoc['inference']
    ), (args.hnmr_file, args.cnmr_file, args.hnmr_shifts, args.cnmr_shifts, args.ckpt, args.normalize)

def main() -> None:
    print("Parsing arguments...")
    args, files = get_args()
    global_args, model_args, inference_args = args

    seed = seed_everything(global_args['seed'])
    dtype = dtype_convert(global_args['dtype'])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not os.path.isdir(global_args['savedir']):
        os.makedirs(global_args['savedir'])

    # Create the dataset
    hnmr_file, cnmr_file, hnmr_shifts, cnmr_shifts, ckpt, normalize = files
    dataset = SingleNMRDataset(hnmr_file=hnmr_file, 
                               cnmr_file=cnmr_file, 
                               hnmr_shifts=hnmr_shifts, 
                               cnmr_shifts=cnmr_shifts, 
                               normalize=normalize,
                               device=device,
                               dtype=dtype)
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    #Construct the model
    model, _ = create_model(model_args, dtype, device)
    model.to(dtype).to(device)
    best_ckpt = torch.load(ckpt, map_location=device)
    model.load_state_dict(best_ckpt['model_state_dict'])

    #Run inference
    predictions = run_inference(model=model,
                                dataloader=dataloader,
                                pred_gen_fn=inference_args['run_inference_args']['pred_gen_fn'],
                                pred_gen_opts=inference_args['run_inference_args']['pred_gen_opts'],
                                write_freq=inference_args['run_inference_args']['write_freq'],
                                device=device)
    
    save_inference_predictions(global_args['savedir'], 
                                train_predictions=None,
                                val_predictions=None,
                                test_predictions=predictions,
                                idx = 0,
                                savetag=f"dataset_0")