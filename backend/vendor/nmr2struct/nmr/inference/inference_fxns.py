import torch
import torch.nn as nn
from typing import Optional
import numpy as np
from torch import Tensor

def infer_basic_model(model: nn.Module, 
                      batch: torch.Tensor, 
                      opts: Optional[dict] = None,
                      device: torch.device = None) -> torch.Tensor:
    """Generate prediction for models that take an input and generate the output in one forward pass
    Args: 
        model: The model to use for inference
        input: The input to the model
        opts: Options to pass to the model as a dictionary, should specify gradient tracking 
            behavior through value 'track_gradients'. If a dictionary is not provided,
            the default behavior is to track gradients
    """
    x, y = batch
    target = y[0]
    #Need additional logic around gradient tracking for modules associated with the 
    #   transformer because it seems behavior can change depending on the no_grad() context
    if opts is None:
        track_gradients = False #Default option
    elif (opts is not None):
        if 'track_gradients' in opts:
            track_gradients = opts['track_gradients']
        else:
            track_gradients = False
    if track_gradients:
        output = model(x)
    else:
        with torch.no_grad():
            output = model(x)
    #Also save x[1] which is the set of SMILES strings
    #   Note that even for a batch size of 1, the batch smiles element
    #   returned by a dataloader is a tuple of the form (str,) which converts
    #   correctly to [str] when using list(). It does not cause the string
    #   to break apart into a list of characters.
    return [(
        target.detach().cpu().numpy(), 
        output.detach().cpu().numpy(),
        list(x[1]),
        np.zeros_like(target.detach().cpu().numpy())
    )]

def infer_multitask_model_substructures(model: nn.Module,
                                        batch: Tensor, 
                                        opts: Optional[dict] = None,
                                        device: torch.device = None) -> Tensor:
    """Generate predictions for the substructure prediction using the multitask model
    Args:
        model: The model to use for inference
        batch: The input to the model
        opts: Options to pass to the model as a dictionary, should specify gradient tracking 
            behavior through value 'track_gradients'. If a dictionary is not provided,
            the default behavior is to track gradients
    """
    x, y = batch
    target = y[1]
    if opts is None:
        track_gradients = False #Default option
    elif (opts is not None):
        if 'track_gradients' in opts:
            track_gradients = opts['track_gradients']
        else:
            track_gradients = False
    if track_gradients:
        output = model(x, y, eval_paths = ['substructure'])
    else:
        with torch.no_grad():
            output = model(x, y, eval_paths = ['substructure'])
    output = output[1].detach().cpu().numpy()
    return [(
        target.detach().cpu().numpy(),
        output,
        list(x[1]),
        np.zeros_like(target.detach().cpu().numpy())
    )]


def get_top_k_sample_batched(k_val: int | float , 
                             character_probabilities: Tensor) -> tuple[Tensor, Tensor]:
    """
    Generates the next character using top-k sampling scheme.

    In top-k sampling, the probability mass is redistributed among the
    top-k next tokens, where k is a hyperparameter. Once redistributed, 
    the next token is sampled from the top-k tokens.
    """
    top_values, top_indices = torch.topk(character_probabilities, k_val, sorted = True)
    #Take the sum of the top probabilities and renormalize
    tot_probs = top_values / torch.sum(top_values, dim = -1).reshape(-1, 1)
    #Sample from the top k probabilities. This represents a multinomial distribution
    try:
        assert(torch.allclose(torch.sum(tot_probs, dim = -1), torch.tensor(1.0)))
    except:
        print("Probabilities did not pass allclose check!")
        print(f"Sum of probs is {torch.sum(tot_probs)}")
    selected_index = torch.multinomial(tot_probs, 1)
    #For gather to work, both tensors have to have the same number of dimensions:
    if len(top_indices.shape) != len(selected_index.shape):
        top_indices = top_indices.reshape(selected_index.shape[0], -1)
    output = torch.gather(top_indices, -1, selected_index)
    output_token_probs = torch.gather(tot_probs, -1, selected_index)
    return output, output_token_probs

def forward_generic_transformer(model: nn.Module,
                                working_x: Tensor,
                                working_y: Tensor,
                                track_gradients: bool = True) -> Tensor:
    """Forward function for infering a generic transformer model
    
    Args:
        model: The transformer model being inferred upon
        working_x: The input to the model
        working_y: The output of the model
        track_gradients: Whether to track gradients during inference. This is because the transformer
            is known to misbehave if gradient tracking is disabled in certain cases
    """
    if track_gradients:
        next_pos = model((working_x, None), (working_y, None)).detach()
    else:
        with torch.no_grad():
            next_pos = model((working_x, None), (working_y, None))
    return next_pos

def forward_multitask_transformer(model: nn.Module,
                                  working_x: Tensor,
                                  working_y: Tensor,
                                  track_gradients: bool = True) -> Tensor:
    """Forward function for infering a multitask transformer model,
    same signature as the generic transformer forward function but 
    the additional model args should be a non-empty dictionary that contains the 
    correct evaluation paths
    """
    if track_gradients:
        next_pos = model((working_x, None), 
                         ((working_y, None), None),
                         eval_paths = ['structure'])
    else:
        with torch.no_grad():
            next_pos = model((working_x, None), 
                             ((working_y, None), None), 
                             eval_paths = ['structure'])
    next_pos = next_pos[0].detach()    
    return next_pos


def infer_transformer_model(model: nn.Module, 
                        batch: torch.Tensor, 
                        opts: dict,
                        device: torch.device = None) -> list[tuple[str, list[str]]] | list[tuple[np.ndarray, list[np.ndarray]]]:
    """Generates a prediction for the input using sampling over a transformer model
    Args:
        model: The model to use for inference
        batch: The input to the model
        opts: Options to pass to the model as a dictionary
        device: The device to use for inference
    
    The opts dictionary should contain the following additional arguments:
        'num_pred_per_tgt' (int): The number of predictions to generate for each input
        'sample_val' (int or float): The sampling value for the model, e.g. the number of values to use for
            top-k sampling
        'stop_token' (int): The stop token to use for the model
        'start_token' (int): The start token to use for the model
        'track_gradients' (bool): Whether to track gradients during inference. This is because the transformer
            is known to misbehave if gradient tracking is disabled in certain cases
        'alphabet' (str): Path to a file containing the alphabet for the model to use in decoding
        'decode' (bool): Whether to decode the output indices of the model against the provided alphabet
        'infer_fwd_fxn' (str): indicator for the forward function to use, one of 'generic', 'multitask'. 
            If not provided, the default is assumed to be 'generic', which maps onto the forward_generic_transformer() function
    """
    x, y = batch
    curr_batch_predictions = []
    curr_batch_sequence_scores = []
    effective_bsize = x[0].shape[0]
    targets = y[1]
    smiles = x[1]
    
    num_pred_per_tgt = opts['num_pred_per_tgt']
    sample_val = opts['sample_val']
    stop_token = opts['tgt_stop_token']
    start_token = opts['tgt_start_token']
    track_gradients = opts['track_gradients']
    alphabet = np.load(opts['alphabet'], allow_pickle=True)
    decode = opts['decode']

    if 'infer_fwd_fxn' not in opts: 
        infer_fwd_fxn = forward_generic_transformer
    else:
        if opts['infer_fwd_fxn'] == 'generic':
            infer_fwd_fxn = forward_generic_transformer
        elif opts['infer_fwd_fxn'] == 'multitask':
            infer_fwd_fxn = forward_multitask_transformer
        else:
            raise ValueError("Invalid forward function specified in opts dictionary")

    for _ in range(num_pred_per_tgt):
        
        #Changing quantities
        working_x = x[0].clone()
        working_y = torch.tensor([start_token] * effective_bsize).reshape(effective_bsize, 1).to(device)
        working_token_probs = torch.tensor([]).to(device)

        #Accumulating quantities
        completed_structures = [None] * effective_bsize
        completed_token_probs = [None] * effective_bsize
        index_mapping = torch.arange(effective_bsize, device = device, dtype = torch.long)
        all_structures_completed = False
        iter_counter = 0

        while not all_structures_completed:
            if (iter_counter % 10 == 0):
                print(f"On iteration {iter_counter}")
            
            next_pos = infer_fwd_fxn(model, working_x, working_y, track_gradients)
            # if track_gradients:
            #     next_pos = model((working_x, None), (working_y, None), **additional_model_args).detach()
            # else:
            #     with torch.no_grad():
            #         next_pos = model((working_x, None), (working_y, None), **additional_model_args)
            
            next_val = next_pos[:, -1, :]
            char_probs = torch.nn.functional.softmax(next_val, dim = -1)
            selected_indices, token_probs = get_top_k_sample_batched(sample_val, char_probs)

            concatenated_results = torch.cat((working_y, selected_indices), dim = -1)
            concatenated_token_probs = torch.cat((working_token_probs, token_probs), dim = -1)

            stop_token_mask = concatenated_results[:, -1] == stop_token
            comp_structs = concatenated_results[stop_token_mask]
            comp_probs = concatenated_token_probs[stop_token_mask]

            comp_inds = index_mapping[stop_token_mask]
            for i, ind in enumerate(comp_inds):
                completed_structures[ind] = comp_structs[i].detach().cpu().numpy()
                completed_token_probs[ind] = comp_probs[i].detach().cpu().numpy()
            
            working_y = concatenated_results[~stop_token_mask]
            working_x = working_x[~stop_token_mask]
            working_token_probs = concatenated_token_probs[~stop_token_mask]
            index_mapping = index_mapping[~stop_token_mask]

            if working_y.shape[-1] > 1000:
                working_y = torch.cat((working_y,
                                       torch.tensor([stop_token] * working_y.shape[0]).reshape(-1, 1).to(device)), 
                                       dim = -1)
                #Add this in so slicing off the stop token is easy
                working_token_probs = torch.cat((working_token_probs,
                                                    torch.tensor([0.0] * working_y.shape[0]).reshape(-1, 1).to(device)),
                                                    dim = -1)
                for j, ind in enumerate(index_mapping):
                    completed_structures[ind] = working_y[j].detach().cpu().numpy()
                    completed_token_probs[ind] = working_token_probs[j].detach().cpu().numpy()
                all_structures_completed = True
            
            if len(working_y) == 0:
                all_structures_completed = True
            
            iter_counter += 1

        for elem in completed_structures:
            assert(elem is not None)
        if decode:
            generated_smiles = []
            for elem in completed_structures:
                try:
                    curr_smi = ''.join(np.array(alphabet)[elem[1:-1].astype(int)])
                    generated_smiles.append(curr_smi)
                except Exception as e:
                    print(e)
                    generated_smiles.append('')
            curr_batch_predictions.append(generated_smiles)
        else:
            curr_batch_predictions.append(completed_structures)
        
        #Compute the scores of the sequences, either SMILES or otherwise. 
        #The scores are computed as the sum of the log probabilities of the tokens 
        #composing the sequence
        scores = []
        for i, elem in enumerate(completed_token_probs):
            assert(len(elem[:-1]) == len(completed_structures[i]) - 2)
            score = np.log(elem[:-1]).sum()
            scores.append(score)
        curr_batch_sequence_scores.append(scores)

    #Final processing
    # print(targets.shape[0])
    # print(len(curr_batch_predictions))
    # print(effective_bsize)
    assert(targets.shape[0] == len(curr_batch_predictions[0]) == effective_bsize)
    generated_predictions = []
    for i in range(effective_bsize):
        #TODO: Think, is this the best way to represent output predictions for each batch as 
        #   tuples of (target, [pred1, pred2, ...])?
        generated_predictions.append((
            smiles[i] if decode else targets[i].detach().cpu().numpy(),
            list(curr_batch_predictions[j][i] for j in range(num_pred_per_tgt)),
            smiles[i],
            list(curr_batch_sequence_scores[j][i] for j in range(num_pred_per_tgt))
        ))
    return generated_predictions