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
        return x, (int(z['sex']), int(z['indian']))
    

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


def test(device, model, test_loader, batch_size, n_batches=None):
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
            
            scores = model(data)
            _, preds = torch.max(scores, 1)
            accuracy = torch.sum(preds == labels).item() / batch_size
            running_acc += accuracy
            
            i += 1
            if i >= n_batches:
                break
    
    return running_acc / n_batches


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

###################
# Create datasets #
###################
# MARK: Create datasets

def create_dataset():
    # Load data
    batch_size = 1000
    _, _, test_loader = load_data(0.5, 0, batch_size)
    
    test_loader_iter = iter(test_loader)
    
    repeat = True
    while repeat:
        repeat = False
        
        samples, (labels, p_targets) = next(test_loader_iter)
        
        american_women_samples = samples[(labels == 0) * (p_targets == 0)][:10]
        indian_women_samples = samples[(labels == 0) * (p_targets == 1)][:10]
        
        american_men_samples = samples[(labels == 1) * (p_targets == 0)][:10]
        indian_men_samples = samples[(labels == 1) * (p_targets == 1)][:10]
        
        all_samples = [american_women_samples,
                       indian_women_samples,
                       american_men_samples,
                       indian_men_samples]
        all_samples = torch.cat(all_samples, dim=0)
        
        if all_samples.size(0) < 40:
            repeat = True
    
    torch.save(all_samples,
               os.path.join(cwd, 'results', '6_privacy', 'fffc_tiny',
                            'attacks_softmax', 'attack_samples.pt'))
    

def save_models_data(model_type='nn',
                     retrained=False,
                     private=False):
    
    print(retrained, private)
    
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    
    aux_mod = import_file('model', os.path.join(cwd, 'models', 'fffc_tiny.py'))
    model_class = aux_mod.Model
    
    # Check tuned config of balanced model
    config_dir = os.path.join(cwd, 'results', '0_train_nns', 
                              model_class.name, '0.5')
    with open(os.path.join(config_dir, 'tuned_config.json'), 'r') as f:
        config = json.load(f)
    
    all_results_dir = os.path.join(cwd, 'results', '6_privacy',
                                   'fffc_tiny', 'attacks_softmax' )
    os.makedirs(all_results_dir, exist_ok=True)
    
    softmax = nn.Softmax(dim=1)
    
    if model_type == 'nn':
        models_dir = os.path.join(cwd, 'results', '0_train_nns', 'fffc_tiny')
        results_dir = os.path.join(all_results_dir, 'nn')
        os.makedirs(results_dir, exist_ok=True)
    
    elif model_type == 'mps':
        
        def embedding(x):
            x = tk.embeddings.poly(x, degree=1)
            return x
        
        if not retrained and not private:
            models_dir = os.path.join(cwd, 'results', '6_privacy',
                                      'fffc_tiny', 'cores')
            results_dir = os.path.join(all_results_dir, 'cores')
            os.makedirs(results_dir, exist_ok=True)
        elif retrained and not private:
            models_dir = os.path.join(cwd, 'results', '6_privacy',
                                      'fffc_tiny', 'recores')
            results_dir = os.path.join(all_results_dir, 'recores')
            os.makedirs(results_dir, exist_ok=True)
        elif retrained and private:
            models_dir = os.path.join(cwd, 'results', '6_privacy',
                                      'fffc_tiny', 'priv_recores')
            results_dir = os.path.join(all_results_dir, 'priv_recores')
            os.makedirs(results_dir, exist_ok=True)
    
    # Load attack samples
    samples_dir = os.path.join(all_results_dir, 'attack_samples.pt')
    if not os.path.exists(samples_dir):
        create_dataset()
    
    samples = torch.load(samples_dir, weights_only=False).to(device)
    
    for p in p_indian_list:
        print(f'{p=}')
        for idx in range(10):
            print(f'\t{idx=}')
            aux_models_dir = os.path.join(models_dir, str(p), str(idx))
            n_models = len(os.listdir(aux_models_dir))
            for s in range(n_models):
                print(f'\t\t{s=}')
                model_dir = list(filter(lambda f: f.startswith(f'{s}_'),
                                        os.listdir(aux_models_dir)))[0]
                
                if model_type == 'nn':
                    state_dict = torch.load(os.path.join(aux_models_dir,
                                                         model_dir),
                                            weights_only=False)
                    
                    model = model_class(config)
                    model.load_state_dict(state_dict)
                    model.eval()
                    model.to(device)
                    
                    results = softmax(model(samples)).cpu()
                
                elif model_type == 'mps':
                    cores = torch.load(os.path.join(aux_models_dir,
                                                    model_dir),
                                       weights_only=False)
                    
                    model = tk.models.MPSLayer(tensors=cores)
                    model.to(device)
                    model.eval()

                    model.trace(torch.zeros(1, out_rate // 2, 2).to(device),
                                inline_input=False,
                                inline_mats=False)
                    
                    results = model(embedding(samples),
                                    inline_input=False,
                                    inline_mats=False).cpu()
                    results = results.pow(2)
                    results = results / results.norm(dim=1, keepdim=True)
                
                results_women = results[:20, 0]
                results_men = results[20:, 1]
                results = torch.cat([results_women, results_men], dim=0)
                
                params = []
                
                for param in model.parameters():
                    params.append(param.flatten().cpu())
                
                params = torch.cat(params, dim=0)
                results = torch.cat([results, params], dim=0)
                
                aux_results_dir = os.path.join(results_dir, str(p), str(idx))
                os.makedirs(aux_results_dir, exist_ok=True)
                
                torch.save(results, os.path.join(aux_results_dir, f'{s}.pt'))


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
        rint('\t<model_type>\n'
             '\t<retrained>\n'
             '\t<private>')
        sys.exit()
      
    # Read options and arguments
    try:
        opts, args = getopt.getopt(argv[1:], 'h', ['help'])
    except getopt.GetoptError:
        print('Available options are:\n'
              '\t--help, -h\n')
        sys.exit(2)

    model_type = None
    retrained = None
    private = None
    if len(args) == 3:
        model_type = args[0]
        retrained = bool(int(args[1]))
        private = bool(int(args[2]))
    else:
        print('All arguments should be passed')
        print('\t<model_type>\n'
              '\t<retrained>\n'
              '\t<private>')
        sys.exit()
        
    save_models_data(model_type=model_type,
                     retrained=retrained,
                     private=private)
