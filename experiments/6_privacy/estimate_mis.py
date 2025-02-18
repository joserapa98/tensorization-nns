# Script used to tensorize trained NN models

import os
import sys
import getopt
import json
from importlib import util

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision
import torchaudio

import tensorkrowch as tk
from tensorkrowch.decompositions import tt_rss

torch.set_num_threads(1)

cwd = os.getcwd()
p_indian_list = [0.005, 0.01, 0.05,
                 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9,
                 0.95, 0.99, 0.995]
out_rate = 1000


def import_file(full_name, path):
    """Returns a python module given its path"""
    spec = util.spec_from_file_location(full_name, path)
    mod = util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#################
# Load Datasets #
#################
# MARK: Load Datasets

class CustomCommonVoice(Dataset):
    """
    Class for the (imbalanced) datasets created.

    Parameters
    ----------
    p_indian : float (p_indian_list)
        Proportion of audios of people with indian accent in the dataset.
    idx : int [0, 9]
        Index of the annotations to be used. For each ``p_indian`` there are 10
        datasets.
    set : str
        Indicates which dataset is to be loaded.
    transform : torchvision.transforms
        Transformations of the dataset (data augmentation, normalization, etc.)
    target_transform : func
        Transformation of the target attribute (not used).
    """
    
    def __init__(self,
                 p_indian,
                 idx,
                 set="train_df.tsv",
                 transform=None):
        
        global p_indian_list
        if (p_indian not in p_indian_list) or ((idx < 0) or (idx > 9)):
            raise ValueError(
                f'`p_indian` can only take values within {p_indian_list}, '
                f'and `idx` should be between 0 and 9')
        
        global cwd
        self.dataset = torchaudio.datasets.COMMONVOICE(
            root=os.path.join(cwd, 'CommonVoice'),
            tsv=os.path.join('datasets', str(p_indian), str(idx), set))
        self.transform = transform

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        x, y, z = self.dataset[index]
        if self.transform:
            x = self.transform((x, y))
        return x, int(z['sex'])
    

def resample(x):
    global out_rate
    x, in_rate = x
    resample_trans = torchaudio.transforms.Resample(in_rate, out_rate)
    return resample_trans(x)

def crop(x):
    global out_rate
    llimit = (x.size(1) // 2 - out_rate // 2)
    rlimit = (x.size(1) // 2 + out_rate // 2)
    x = x[:, llimit:rlimit].flatten()
    if x.size(0) < out_rate:
        return None
    return x

def rfft(x):
    if x is None:
        return None
    return torch.fft.rfft(x)[:-1].abs()

def normalize(x):
    x = x / 200
    x = torch.where(x <= 0, 1e-5, x)
    x = torch.where(x >= 1, 1 - 1e-5, x)
    return x

transform = torchvision.transforms.Compose([
    torchvision.transforms.Lambda(resample),
    torchvision.transforms.Lambda(crop),
    torchvision.transforms.Lambda(rfft),
    torchvision.transforms.Lambda(normalize)
    ])

def none_collate(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def load_data(p_indian, idx, batch_size):
    """Loads dataset performing the required transformations for train or test."""
    
    # Load datasets
    global transform
    train_dataset = CustomCommonVoice(p_indian,
                                      idx,
                                      set="train_df.tsv",
                                      transform=transform)
    val_dataset = CustomCommonVoice(p_indian,
                                    idx,
                                    set="val_df.tsv",
                                    transform=transform)
    test_dataset = CustomCommonVoice(p_indian,
                                     idx,
                                     set="test_df.tsv",
                                     transform=transform)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              collate_fn=none_collate,
                              shuffle=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            collate_fn=none_collate,
                            shuffle=False)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             collate_fn=none_collate,
                             shuffle=False)
    
    return train_loader, val_loader, test_loader


def load_sketch_samples(p_indian, idx, batch_size):
    """Loads sketch samples to tensorize models."""
    
    # Load datasets
    global transform
    test_tensorize_dataset = CustomCommonVoice(p_indian,
                                               idx,
                                               set="test_df_tensorize.tsv",
                                               transform=transform)
    test_unused_dataset = CustomCommonVoice(p_indian,
                                            idx,
                                            set="test_df_unused.tsv",
                                            transform=transform)
    
    # Create DataLoaders
    test_tensorize_loader = DataLoader(test_tensorize_dataset,
                                       batch_size=500,
                                       collate_fn=none_collate,
                                       shuffle=False)
    test_unused_loader = DataLoader(test_unused_dataset,
                                    batch_size=batch_size,
                                    collate_fn=none_collate,
                                    shuffle=False)
    
    return test_tensorize_loader, test_unused_loader


def training_epoch_tn(device, model, embedding, criterion, optimizer,
                      train_loader, n_batches=None):
    if n_batches is not None:
        n_batches = min(n_batches, len(train_loader))
    else:
        n_batches = len(train_loader)
    i = 0

    model.train()
    for data, labels in train_loader:
        data = data.to(device)
        labels = labels.to(device)
        
        # Forward
        scores = model(embedding(data),
                       inline_input=False,
                       inline_mats=False)
        scores = scores.pow(2)
        scores = scores / scores.norm(dim=1, keepdim=True)
        
        loss = criterion(scores, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient descent
        optimizer.step()
        
        i += 1
        if i >= n_batches:
            break
    
    return model


def test_tn(device, model, embedding, test_loader, batch_size, n_batches=None):
    """Computes accuracy on test set."""
    running_acc = 0
    
    if n_batches is not None:
        n_batches = min(n_batches, len(test_loader))
    else:
        n_batches = len(test_loader)
    i = 0
    
    model.eval()
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.to(device)
            
            scores = model(embedding(data),
                           inline_input=False,
                           inline_mats=False)
            scores = scores.pow(2)
            scores = scores / scores.norm(dim=1, keepdim=True)
            
            _, preds = torch.max(scores, 1)
            accuracy = torch.sum(preds == labels).item() / batch_size
            running_acc += accuracy
            
            i += 1
            if i >= n_batches:
                break
    
    return running_acc / n_batches


###############################################################################
###############################################################################

######################
# Mutual Information #
######################
# MARK: Mutual Information

@torch.no_grad()
def estimate_mi(fun, cut, samples, device):
    """
    Estimates MI of function fun using only points in dataset.

    Parameters
    ----------
    fun : function
        Function returning values in [0, 1].
    cut : int
        Indices < cut constitute the left subsystem, indices >= cut
        constitute the right subsystem. MI is computed for the correlation
        between left and right subsystems.
    samples : torch.Tensor
        Samples used to estimate MI. They include labels. It has shape
        ``batch x (in_dim + 1)``
    """
    samples = samples.to(device)
        
    # Obtain unique left and right parts of samples in dataset
    left = samples[:, :cut].unique(dim=0)  # batch x dim
    right = samples[:, cut:].unique(dim=0) # batch x dim
    
    # Create mesh grid where each row shares the same left part,
    # and each column shares the same right part in the samples
    left_aux = left.unsqueeze(1).expand(-1, right.size(0), -1)
    right_aux = right.unsqueeze(0).expand(left.size(0), -1, -1)

    mesh = torch.cat([left_aux, right_aux], dim=2)
    
    # Evaluate fun on all points of the grid
    values = fun(mesh.view(left.size(0) * right.size(0), -1))
    values = values.view(mesh.size(0), mesh.size(1))
    values = values / values.sum()  # Normalize to have distribution
    
    # Sum values in each row (share left part) and each column (share right part)
    partials_left = values.sum(1)
    partials_right = values.sum(0)
    
    # Outer product of partials_left and partials_right
    cross_values = torch.outer(partials_left, partials_right)
    
    # Compute MI
    logs = torch.log(torch.div(values, cross_values))
    logs = torch.where(torch.logical_or(logs.isnan(), logs.isinf()), 0., logs)
    mi = torch.sum(torch.mul(values, logs))
    
    return mi


def estimate_mi_all_cuts_aux(fun, samples, device):
    lst_mis = []
    # Dim of vectors in dataset is equal to number of sites in MPS
    for cut in range(1, samples.size(1)): 
        lst_mis.append(estimate_mi(fun, cut, samples, device))
    return torch.stack(lst_mis, dim=0)


def estimate_mi_all_cuts(fun, samples, device):
    mi_tensor = estimate_mi_all_cuts_aux(fun, samples, device)
    return torch.stack([mi_tensor.mean(),
                        mi_tensor.std(),
                        mi_tensor.max()]).cpu()


###############################################################################
###############################################################################

########
# MI's #
########
# MARK: MI's

def estimate_mi_nns(model_class, p_indian):
    """Tensorizes models trained with the given ``p_indian`` and ``idx``."""
    global p_indian_list
    
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    
    if p_indian is not None:
        p_indian_list_aux = [p_indian]
    else:
        p_indian_list_aux = p_indian_list
        
    idx_range_aux = range(10)
    
    # Check tuned config of balanced model
    config_dir = os.path.join(cwd, 'results', '0_train_nns',
                              model_class.name, '0.5')
    with open(os.path.join(config_dir, 'tuned_config.json'), 'r') as f:
        config = json.load(f)
    
    softmax = nn.Softmax(dim=1)
        
    for p_indian in p_indian_list_aux:
        for idx in idx_range_aux:
            models_dir = os.path.join(cwd, 'results', '0_train_nns',
                                      model_class.name, str(p_indian), str(idx))
            sketches_dir = os.path.join(cwd, 'results', '6_privacy',
                                        model_class.name, 'sketches',
                                        str(p_indian), str(idx))
            
            mis_dir = os.path.join(cwd, 'results', '6_privacy',
                                   model_class.name, 'mis')
            os.makedirs(mis_dir, exist_ok=True)
            
            n_models = len(os.listdir(models_dir))
            
            mis_tensor = torch.empty(10, n_models, 3, dtype=torch.float32)
            
            for s in range(n_models):
                # Load state_dict
                state_dict_dir = list(filter(lambda f: f.startswith(f'{s}_'),
                                             os.listdir(models_dir)))[0]
                state_dict = torch.load(os.path.join(models_dir, state_dict_dir),
                                        weights_only=False)
                
                # Initialize model with balanced config
                model = model_class(config)
                model.load_state_dict(state_dict)
                model.eval()
                model.to(device)
                
                # Load sketch samples
                sketch_samples, sketch_labels = torch.load(
                    os.path.join(sketches_dir, f'{s}.pt'),
                    weights_only=False
                    )
                
                # n_samples = 100
                n_features = sketch_samples.shape[1] + 1
                out_position = n_features // 2
                
                # Check MI of NN
                def mi_fn(samples):
                    labels = samples[:, out_position:(out_position + 1)].to(torch.int64)
                    samples = torch.cat([samples[:, :out_position],
                                         samples[:, (out_position + 1):]],
                                        dim=1)
                    result = softmax(model(samples))
                    outputs = result.gather(dim=1, index=labels).flatten()
                    return outputs
                
                mi_samples = torch.cat([sketch_samples[:, :out_position],
                                        sketch_labels.unsqueeze(1).float(),
                                        sketch_samples[:, out_position:]],
                                        dim=1)#[:n_samples]
                
                aux_mi_tensor = estimate_mi_all_cuts(mi_fn, mi_samples, device)
                max_bond_dim = aux_mi_tensor[2].exp()
                
                # Print logs
                print(f'**{model_class.name}** (p: {p_indian}, i: {idx}, s: {s}) => '
                      f'MI: mean={aux_mi_tensor[0].item():.4f} / '
                      f'std={aux_mi_tensor[1].item():.4f} / '
                      f'max={aux_mi_tensor[2].item():.4f} / '
                      f'{max_bond_dim=:.4}'
                      )
                
                mis_tensor[idx, s, :] = aux_mi_tensor
        
        torch.save(mis_tensor, os.path.join(mis_dir, f'{p_indian}.pt'))


###############################################################################
###############################################################################

###############
# System args #
###############
# MARK: System args

if __name__ == '__main__':
    argv = sys.argv
    if len(argv) == 1:
        print('No argumets were passed')
        print('Available options are:\n'
              '\t--help, -h\n'
              '\t<model name>\n'
              f'\t(optional) proportion of imbalance (p in {p_indian_list})')
        sys.exit()
      
    # Read options and arguments
    try:
        opts, args = getopt.getopt(argv[1:], 'h', ['help'])
    except getopt.GetoptError:
        print('Available options are:\n'
              '\t--help, -h\n')
        sys.exit(2)

    model_name = None
    p_indian = None
    if len(args) == 1:
        model_name = args[0]
    elif len(args) == 2:
        model_name = args[0]
        p_indian = float(args[1])
        
    aux_mod = import_file('model', os.path.join(cwd, 'models', f'{model_name}.py'))
    model_class = aux_mod.Model
    estimate_mi_nns(model_class=model_class, p_indian=p_indian)
