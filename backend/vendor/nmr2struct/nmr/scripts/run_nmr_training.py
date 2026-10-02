import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import yaml
import os
from nmr.data import create_dataset
from nmr.models import create_model
from nmr.training import create_optimizer, fit
import nmr.training.loss_fxns as loss_fxns
import h5py
import pickle as pkl
from .top_level_utils import (
    seed_everything, 
    seed_worker, 
    dtype_convert, 
    save_completed_config, 
    split_data_subsets,
    save_train_history,
    save_token_size_dict,
    specific_update
)

def get_args() -> dict:
    '''Parses the passed yaml file to get arguments'''
    parser = argparse.ArgumentParser(description='Run NMR training')
    parser.add_argument('config_file', type = str, help = 'The path to the YAML configuration file')
    args = parser.parse_args()
    listdoc =  yaml.safe_load(open(args.config_file, 'r'))
    return (
        listdoc['global_args'],
        listdoc['data'],
        listdoc['model'],
        listdoc['training']
    )

def main() -> None:
    # view towards hydra 
    # argparse for now
    # Separate arguments
    print("Parsing arguments...")
    global_args, dataset_args, model_args, training_args = get_args()

    # Set up consistent device, datatype, and seed
    print("Setting up device, datatype, and seed...")
    device = torch.device('cuda:0' if global_args['ngpus'] > 0 else 'cpu')
    dtype = dtype_convert(global_args['dtype'])
    seed = seed_everything(global_args['seed'])

    if not os.path.isdir(global_args['savedir']):
        os.makedirs(global_args['savedir'])

    print("Initializing dataset, model, optimizer, loss, and scheduler...")
    # Set up dataset, model, optimizer, loss, and scheduler
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

    # print(f"Generated {len(datasets)} datasets")
    # dataset.save_smiles_alphabet(global_args['savedir'])
    # size_dict = dataset.get_sizes()
    # token_dict = dataset.get_ctrl_tokens()
    # max_len_dict = dataset.get_max_seq_len()
    # total_dict = {**size_dict, **token_dict, **max_len_dict}

    #Fix target pad token as ignore index
    tgt_pad_token = total_dict['tgt_pad_token']
    total_dict['ignore_index'] = tgt_pad_token if tgt_pad_token is not None else -100
    total_dict['seed'] = seed
    print(total_dict)

    #Update model args
    model_args = specific_update(model_args, total_dict)
    #Update training args
    training_args = specific_update(training_args, total_dict)

    for i, d in enumerate(datasets):
        print(f"Max sequence length for dataset {i}: {d.get_max_seq_len()}")

    model, updated_model_args = create_model(model_args, dtype, device)
    model.to(dtype).to(device)
    
    print("Total number of trainable parameters", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(model)

    optimizer = create_optimizer(model, updated_model_args, training_args, dtype, device)
    loss_fn = getattr(loss_fxns, training_args['loss_fn'])

    if training_args['loss_fn_args'] is not None:
        loss_fn = loss_fn(**training_args['loss_fn_args'])
    else:
        loss_fn = loss_fn()
    
    if hasattr(loss_fn, 'ignore_index'):
        print(f"Setting ignore index for loss function to {loss_fn.ignore_index}")

    if training_args['scheduler'] is not None:
        scheduler_raw = getattr(optim.lr_scheduler, training_args['scheduler'])
        scheduler = scheduler_raw(optimizer, **training_args['scheduler_args'])
    else:
        scheduler = None

    # Set up dataloaders
    print("Setting up dataloaders...")
    #Load in multiple splits (now also a list)
    splits = training_args['splits']
    assert len(splits) == len(datasets)
    train_sets, val_sets, test_sets = [], [], []
    for i, s in enumerate(splits):
        train_set, val_set, test_set = split_data_subsets(datasets[i], 
                                                        s,
                                                        training_args['train_size'],
                                                        training_args['val_size'],
                                                        training_args['test_size'])
        train_sets.append(train_set)
        val_sets.append(val_set)
        test_sets.append(test_set)

    # Set up seeding in accordance with https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    g = torch.Generator()
    g.manual_seed(0)

    train_loaders, val_loaders, test_loaders = [], [], []

    # import pdb; pdb.set_trace()

    for i in range(len(train_sets)):
        if len(train_sets[i]) > 0:
            train_loader = DataLoader(train_sets[i], worker_init_fn=seed_worker, generator=g, **training_args['dloader_args'])
        else:
            train_loader = []
        
        if len(val_sets[i]) > 0:
            val_loader = DataLoader(val_sets[i], worker_init_fn=seed_worker, generator=g, **training_args['dloader_args'])
        else:
            val_loader = []
        
        if len(test_sets[i]) > 0:
            test_loader = DataLoader(test_sets[i], worker_init_fn=seed_worker, generator=g, **training_args['dloader_args'])
        else:
            test_loader = []

        train_loaders.append(train_loader)
        val_loaders.append(val_loader)
        test_loaders.append(test_loader)

    # train_loader = DataLoader(train_set, worker_init_fn=seed_worker, generator=g, **training_args['dloader_args'])
    # val_loader = DataLoader(val_set, worker_init_fn=seed_worker, generator=g, **training_args['dloader_args'])
    # test_loader = DataLoader(test_set, worker_init_fn=seed_worker, generator=g, **training_args['dloader_args'])

    # Set up tensorboard writer
    writer = SummaryWriter(log_dir = global_args['savedir'])

    # Save completed config
    tot_config = {
        'global_args' : global_args,
        'data' : updated_dataset_args,
        'model' : updated_model_args,
        'training' : training_args
    }
    save_completed_config('full_train_config.yaml', tot_config, global_args['savedir'])
    save_token_size_dict(global_args['savedir'], total_dict, 'train')

    # Train
    #TODO: Update the code for the fitter to handle multiple datasets
    print("Beginning training")
    losses = fit(model=model,
                train_dataloaders=train_loaders,
                val_dataloaders=val_loaders,
                test_dataloaders=test_loaders,
                loss_fn=loss_fn,
                optimizer=optimizer,
                nepochs=training_args['nepochs'],
                save_dir=global_args['savedir'],
                writer=writer,
                scheduler=scheduler,
                top_checkpoints_n=training_args['top_checkpoints_n'],
                loss_metric=training_args['checkpoint_loss_metric'],
                write_freq=training_args['write_freq'],
                test_freq=training_args['test_freq'],
                prev_epochs=training_args['prev_epochs']
                )
    
    # save_train_history(global_args['savedir'], losses)
    with open(f"{global_args['savedir']}/losses.pkl", "wb") as f:
        pkl.dump(losses, f)

if __name__ == '__main__':
    main()
