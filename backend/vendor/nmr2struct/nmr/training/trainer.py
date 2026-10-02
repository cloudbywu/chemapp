import numpy as np
import pickle, os, shutil
import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable, Optional
import re

def train_loop(model: nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: Callable[[Tensor, Tensor], Tensor], 
               optimizer: torch.optim.Optimizer, 
               epoch: int, 
               writer: torch.utils.tensorboard.SummaryWriter, 
               scheduler: Optional[torch.optim.lr_scheduler.LambdaLR], 
               write_freq: int = 100,
               write_tag: str = "") -> float:
    """Model training loop
    Args:
        model: The model to train
        dataloader: The dataloader for the training dataset
        loss_fn: The loss function to use for the model, with the signature
            tensor, tensor -> tensor
        optimizer: The optimizer for training the model
        epoch: The current epoch
        writer: Tensorboard writer for logging losses and learning rates
        scheduler: The optional learning rate scheduler
        write_freq: The frequency for printing loss information
        write_tag: For naming unique losses in the tensorboard writer
    """
    tot_loss = 0
    model.train()
    for ibatch, (x, y) in enumerate(dataloader):
        inner_step = int(( epoch * len(dataloader)) + ibatch)
        loss = model.get_loss(x,y,loss_fn) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #Step the learning rate scheduler too based on the current optimizer step
        if scheduler is not None:
            scheduler.step()
        if (ibatch % write_freq == 0):
            print(f"Epoch: {epoch}\tBatch:{ibatch}\tTrain Loss{write_tag}: {loss.item()}")
        tot_loss += loss.item()
        writer.add_scalar(f"Training Step Loss{write_tag}", loss.item(), inner_step)
  
    if len(dataloader) > 0:
        writer.add_scalar(f"Avg. Epoch Train Loss{write_tag}", tot_loss / len(dataloader), epoch)
        return tot_loss / len(dataloader)   
    else:
        writer.add_scalar(f"Avg. Epoch Train Loss{write_tag}", 0, epoch)
        return 0
   

def validation_loop(model: nn.Module, 
                    dataloader: torch.utils.data.DataLoader, 
                    loss_fn: Callable[[Tensor, Tensor], Tensor], 
                    epoch: int, 
                    writer: torch.utils.tensorboard.SummaryWriter, 
                    write_freq: int = 100,
                    write_tag: str="") -> float:
    """Model validation loop
    Args:
        model: The model to validate
        dataloader: The dataloader for the validation dataset
        loss_fn: The loss function to use for the model, with the signature
            tensor, tensor -> tensor
        epoch: The current epoch
        writer: Tensorboard writer for logging losses and learning rates
        write_freq: The frequency for printing loss information
        write_tag: For naming unique losses in the tensorboard writer
    """
    tot_loss = 0
    model.eval()
    for ibatch, (x, y) in enumerate(dataloader):
        inner_step = int(( epoch * len(dataloader)) + ibatch)
        loss = model.get_loss(x,y,loss_fn).detach()
        if (ibatch % write_freq == 0):
            print(f"Epoch: {epoch}\tBatch:{ibatch}\tValidation Loss{write_tag}: {loss.item()}")
        tot_loss += loss.item()
        writer.add_scalar(f"Validation Step Loss{write_tag}", loss.item(), inner_step)
    
    if len(dataloader) > 0:
        writer.add_scalar(f"Avg. Epoch Validation Loss{write_tag}", tot_loss / len(dataloader), epoch)
        return tot_loss / len(dataloader)
    else:
        writer.add_scalar(f"Avg. Epoch Validation Loss{write_tag}", 0, epoch)
        return 0


def test_loop(model: nn.Module, 
                dataloader: torch.utils.data.DataLoader, 
                loss_fn: Callable[[Tensor, Tensor], Tensor], 
                epoch: int, 
                writer: torch.utils.tensorboard.SummaryWriter, 
                write_freq: int = 100,
                write_tag: str="") -> float:
    """Model test loop
    Args:
        model: The model to test
        dataloader: The dataloader for the test dataset
        loss_fn: The loss function to use for the model, with the signature
            tensor, tensor -> tensor
        epoch: The current epoch
        writer: Tensorboard writer for logging losses and learning rates
        write_freq: The frequency for printing loss information
        write_tag: For naming unique losses in the tensorboard writer
    """
    tot_loss = 0
    model.eval()
    for ibatch, (x, y) in enumerate(dataloader):
        inner_step = int(( epoch * len(dataloader)) + ibatch)
        loss = model.get_loss(x,y,loss_fn).detach()
        if (ibatch % write_freq == 0):
            print(f"Epoch: {epoch}\tBatch:{ibatch}\tTest Loss{write_tag}: {loss.item()}")
        tot_loss += loss.item()
        writer.add_scalar(f"Test Step Loss{write_tag}", loss.item(), inner_step)
    
    if len(dataloader) > 0:
        writer.add_scalar(f"Avg. Epoch Test Loss{write_tag}", tot_loss / len(dataloader), epoch)
        return tot_loss / len(dataloader)
    else:
        writer.add_scalar(f"Avg. Epoch Test Loss{write_tag}", 0, epoch)
        return 0


def save_model(model: nn.Module, 
               optim: torch.optim.Optimizer, 
               scheduler: Optional[torch.optim.lr_scheduler.LambdaLR],
               epoch: int, 
               loss_metric: float, 
               savedir: str, 
               savename: str = None) -> str:
    """Save model and optimizer state dicts to file
    Args:
        model: The model to save
        optim: The optimizer to save
        scheduler: The scheduler to save (can be None)
        epoch: The current epoch
        loss_metric: The loss value to associate with the 
        savedir: The directory to save the model and optimizer state dicts
        savename: The name to save for the checkpoint. If None, then the default format
            for saving models is used: model_epoch={epoch}_loss={loss_metric:.8f}.pt
    """
    if savename is None:
        savename = f'{savedir}/model_epoch={epoch}_loss={loss_metric:.8f}.pt'
    else:
        savename = savename
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'scheduler_state_dict' : scheduler.state_dict() if scheduler is not None else None,
                'epoch' : epoch},
                savename)
    return savename
    
def delete_checkpoint(checkpoint: str) -> None:
    """Delete a checkpoint file
    Args:
        checkpoint: The path to the checkpoint file
    """
    os.remove(checkpoint)

def extract_loss_val(checkpoint_name: str) -> float:
    """Extract loss value from a checkpoint name"""
    return float(re.findall(r"\d+\.\d+", checkpoint_name)[0])

def determine_existing_checkpoints(savedir: str) -> tuple[list, list]:
    all_files = os.listdir(savedir)
    checkpoint_files = list(filter(lambda x: x.endswith('.pt') and x != "RESTART_checkpoint.pt", all_files))
    if len(checkpoint_files) == 0:
        return [], []
    else:
        checkpoint_losses = list(map(extract_loss_val, checkpoint_files))
        checkpoint_files = list(map(lambda x: f'{savedir}/{x}', checkpoint_files))
        return checkpoint_files, checkpoint_losses

def fit(model: nn.Module,
        train_dataloaders: torch.utils.data.DataLoader, 
        val_dataloaders: torch.utils.data.DataLoader, 
        test_dataloaders: torch.utils.data.DataLoader, 
        loss_fn: Callable[[Tensor, Tensor], Tensor], 
        optimizer: torch.optim.Optimizer, 
        nepochs: int,
        save_dir: str,
        writer: torch.utils.tensorboard.SummaryWriter, 
        scheduler: Optional[torch.optim.lr_scheduler.LambdaLR], 
        top_checkpoints_n: int = 10,
        loss_metric: str = 'val',
        write_freq: int = 100,
        test_freq: int = 10,
        prev_epochs: int = 0) -> tuple[list, list, list, list]:
    """Model training loop

    Args:
        model: The model to train
        train_dataloaders: A list of the dataloaders for the training datasets
        val_dataloaders: A list of the dataloaders for the validation datasets
        test_dataloaders: A list of the dataloader for the test datasets
        loss_fn: The loss function to use for the model, with the signature
            tensor, tensor -> tensor
        optimizer: The optimizer for training the model
        nepochs: The number of epochs to train for
        save_dir: The directory to save the model and optimizer state dicts
        writer: Tensorboard writer for logging losses and learning rates
        scheduler: The optional learning rate scheduler
        top_checkpoints_n: The number of top checkpoints to save
        loss_metric: The criterion to use for saving checkpoints. Can be 'val' or 'train'
        write_freq: The frequency for printing loss information
        test_freq: The frequency for running the test loop of the model
        prev_epochs: The number of epochs that have already been trained for. This is used
            for loading checkpoints
    """
    assert len(train_dataloaders) == len(val_dataloaders) == len(test_dataloaders)
    existing_files, existing_losses = determine_existing_checkpoints(save_dir)
    assert(len(existing_files) == len(existing_losses))
    best_losses = np.concatenate((np.ones(top_checkpoints_n - len(existing_losses)) * np.inf, np.array(existing_losses)))
    model_names = [None] * (top_checkpoints_n - len(existing_files)) + existing_files
    train_metrics = {
        f'train_loss_{x}' : [] for x in range(len(train_dataloaders))
    }
    val_metrics = {
        f'val_loss_{x}' : [] for x in range(len(val_dataloaders))
    }
    test_metrics = {
        f'test_loss_{x}' : [] for x in range(len(test_dataloaders))
    }
    for epoch in range(nepochs):
        true_epoch = epoch + prev_epochs

        #Train loss computations
        for i_train, train_dloader in enumerate(train_dataloaders):
            train_loss = train_loop(model, 
                                    train_dloader, 
                                    loss_fn, 
                                    optimizer, 
                                    true_epoch, 
                                    writer, 
                                    scheduler, 
                                    write_freq,
                                    write_tag=f" {i_train}")
            train_metrics[f'train_loss_{i_train}'].append(train_loss)
        #Validation loss computations
        for i_val, val_dloader in enumerate(val_dataloaders):
            val_loss = validation_loop(model, 
                                    val_dloader, 
                                    loss_fn, 
                                    true_epoch, 
                                    writer, 
                                    write_freq,
                                    write_tag = f" {i_val}")
            val_metrics[f"val_loss_{i_val}"].append(val_loss)
        if true_epoch % test_freq == 0:
            #Test loss calculation
            for i_tst, tst_dloader in enumerate(test_dataloaders):
                test_loss = test_loop(model,
                                    tst_dloader,
                                    loss_fn,
                                    true_epoch,
                                    writer,
                                    write_freq,
                                    write_tag = f" {i_tst}")
                test_metrics[f"test_loss_{i_tst}"].append(test_loss)
        
        if 'train' in loss_metric:
            curr_k_metric_value = train_metrics[loss_metric][-1]
        elif 'val' in loss_metric:
            curr_k_metric_value = val_metrics[loss_metric][-1]
        # elif 'test' in loss_metric:
        #     curr_k_metric_value = test_metrics[loss_metric][-1]
        else:
            raise ValueError("Invalid monitoring metric")

        max_loss_idx = np.argmax(best_losses)
        max_loss_value = best_losses[max_loss_idx]
        max_loss_model = model_names[max_loss_idx]
        if curr_k_metric_value < max_loss_value:
            if max_loss_model is not None:
                delete_checkpoint(max_loss_model)
            #Set the savename to None here to save the model checkpoints
            #   using the default filename format
            model_name = save_model(model, 
                                    optimizer, 
                                    scheduler,
                                    true_epoch, 
                                    curr_k_metric_value, 
                                    save_dir,
                                    savename=None)
            #Update loss value and model names
            best_losses[max_loss_idx] = curr_k_metric_value
            model_names[max_loss_idx] = model_name
        #Save a restart model every epoch in case the training crashes
        #   or needs to be restarted
        save_model(model,
                   optimizer,
                   scheduler,
                   true_epoch,
                   curr_k_metric_value,
                   save_dir,
                   savename = f"{save_dir}/RESTART_checkpoint.pt")
    writer.flush()
    writer.close()

    for i, tst_dloader in enumerate(test_dataloaders):
        final_test_loss = test_loop(model,
                                    tst_dloader,
                                    loss_fn,
                                    true_epoch,
                                    writer,
                                    write_freq,
                                    write_tag = f" {i}")
        test_metrics[f'test_loss_{i}'].append(final_test_loss)
    
    if nepochs >= top_checkpoints_n:
        assert(None not in model_names)
        assert(all(best_losses < np.inf))
    return train_metrics, val_metrics, test_metrics, model_names, best_losses