import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import argparse
import yaml
from nmr.data import create_dataset
from nmr.models import create_model
from nmr.training import create_optimizer, fit
import nmr.training.loss_fxns as loss_fxns
import os
import hydra
from nmr.scripts.top_level_utils import (
    seed_everything,
    seed_worker,
    dtype_convert,
    save_completed_config,
    split_data_subsets,
    save_train_history,
    save_token_size_dict,
    specific_update
)
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from nmr.models.lightning_module import LightningModel
from omegaconf import OmegaConf
import torch
from lightning.pytorch.strategies import FSDPStrategy

def get_args() -> dict:
    '''Parses the passed yaml file to get arguments'''
    parser = argparse.ArgumentParser(description='Run NMR training with lightning enabled')
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
    print("Parsing arguments...")
    global_args, dataset_args, model_args, training_args = get_args()
    dtype, device = dtype_convert(global_args['dtype']), None #Device handled by ptl trainer

    dataset, updated_dataset_args = create_dataset(dataset_args, dtype, device)
    size_dict = dataset.get_sizes()
    token_dict = dataset.get_ctrl_tokens()
    max_len_dict = dataset.get_max_seq_len()
    total_dict = {**size_dict, **token_dict, **max_len_dict}
    # Fix target pad token as ignore index
    tgt_pad_token = total_dict['tgt_pad_token']
    total_dict['ignore_index'] = tgt_pad_token if tgt_pad_token is not None else -100
    total_dict['seed'] = global_args['seed']

    print("Setting up dataloaders...")
    train_set, val_set, test_set = split_data_subsets(dataset,
                                                      training_args['splits'],
                                                      training_args['train_size'],
                                                      training_args['val_size'],
                                                      training_args['test_size'])
    
    g = torch.Generator()
    g.manual_seed(0)

    train_loader = DataLoader(train_set, worker_init_fn=seed_worker, generator=g, shuffle=True,
                              **training_args['dloader_args'])
    val_loader = DataLoader(val_set, worker_init_fn=seed_worker,
                            generator=g, shuffle=False, **training_args['dloader_args'])
    test_loader = DataLoader(test_set, worker_init_fn=seed_worker,
                             generator=g, shuffle=False, **training_args['dloader_args'])

    # Update model args
    model_args = specific_update(model_args, total_dict)
    # Update training args
    training_args = specific_update(training_args, total_dict)

    L.seed_everything(global_args['seed'])
    savedir = global_args['savedir']

    every_epoch_checkpoint_callback = ModelCheckpoint(filename="model_{epoch:02d}_loss={validation_loss:.10f}",
                                                      monitor='validation_loss',
                                                      mode='min',
                                                      every_n_epochs=1,
                                                      save_top_k=training_args['top_checkpoints_n'])
    restart_checkpoint_callback = ModelCheckpoint(filename="RESTART_checkpoint", 
                                                  every_n_epochs=1,
                                                  save_top_k=1)
    
    
    logger = TensorBoardLogger(save_dir = "./")
    lightning_model = LightningModel(
        model_args = model_args,
        training_args = training_args
    )

    if (global_args['ngpus'] != 0):
        if training_args['strategy'] == 'fsdp':
            strategy = FSDPStrategy(
                sharding_strategy="FULL_SHARD"
            )
        else:
            strategy = training_args['strategy']
        trainer = L.Trainer(
            max_epochs = training_args['nepochs'],
            logger = logger,
            callbacks = [every_epoch_checkpoint_callback, restart_checkpoint_callback],
            accelerator='cuda',
            devices=global_args['ngpus'],
            strategy=strategy,
            inference_mode=False, #Critical to allow gradients in validation and testing steps
            precision="16-mixed"
        )
    else:
        trainer = L.Trainer(
            max_epochs = training_args['nepochs'],
            logger = logger,
            callbacks = [every_epoch_checkpoint_callback, restart_checkpoint_callback],
            accelerator = 'cpu',
            inference_mode=False,
            precision="16-mixed"
        )

    if trainer.global_rank == 0:
        train_folder_name = trainer.logger.log_dir
        os.makedirs(train_folder_name, exist_ok = True)
        tot_config = {
            'global_args' : global_args,
            'data' : updated_dataset_args,
            'model' : model_args,
            'training' : training_args
        }
        OmegaConf.save(tot_config, f"{train_folder_name}/config.yaml")
        dataset.save_smiles_alphabet(train_folder_name)
    
    trainer.fit(lightning_model, train_loader, val_loader, 
                ckpt_path = training_args['restart_ckpt'] if training_args['restart_ckpt'] is not None else None)
    
    #No testing! Breaks things, e.g. see https://github.com/pytorch/pytorch/issues/100069
    # trainer.test(lightning_model, test_loader)