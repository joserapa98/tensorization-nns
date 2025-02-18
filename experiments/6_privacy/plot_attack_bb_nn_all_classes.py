import os
import sys
import getopt
import json
from importlib import util
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision
import torchaudio

torch.set_num_threads(1)

cwd = os.getcwd()
p_indian_list = [0.005, 0.01, 0.05,
                 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
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
                 sex,
                 indian,
                 transform=None):
        
        global cwd
        self.dataset = torchaudio.datasets.COMMONVOICE(
            root=os.path.join(cwd, 'CommonVoice'),
            tsv=os.path.join('datasets', 'full_df_test_copy.tsv'))
        self.transform = transform
        
        self.sex = sex
        self.canadian = indian

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        x, y, z = self.dataset[index]
        if self.transform:
            x = self.transform((x, y))
            
        if (int(z['sex']) != self.sex) or (int(z['canadian']) != self.canadian):
            return None, None
        
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


def load_data(sex, indian, batch_size):
    """Loads dataset performing the required transformations for train or test."""
    
    # Load datasets
    global transform
    test_dataset = CustomCommonVoice(sex,
                                     indian,
                                     transform=transform)
    
    # Create DataLoaders
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             collate_fn=none_collate,
                             shuffle=True)
    
    return test_loader


def test(device, model, test_loader, n_batches=None):
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
            accuracy = (preds == labels).float().mean().item()
            running_acc += accuracy
            
            i += 1
            if i >= n_batches:
                break
    
    return running_acc / n_batches


def accs_by_class(sex, indian):
    model_name = 'fffc_tiny'

    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    aux_mod = import_file('model', os.path.join(cwd, 'models', f'{model_name}.py'))
    model_class = aux_mod.Model

    # Check tuned config of balanced model
    config_dir = os.path.join(cwd, 'results', '0_train_nns',
                              model_class.name, '0.5')
    with open(os.path.join(config_dir, 'tuned_config.json'), 'r') as f:
        config = json.load(f)

    # Load data
    batch_size = 1000
    test_loader = load_data(sex, indian, batch_size)

    for p_indian in p_indian_list:
        print(f'{p_indian=}')
        p_accs = []
        for idx in range(10):
            print(f'\t{idx=}')
            models_dir = os.path.join(cwd, 'results', '0_train_nns',
                                      model_class.name, str(p_indian), str(idx))
            if os.path.exists(models_dir):
                n_models = len(os.listdir(models_dir))
                for s in range(n_models):
                    print(f'\t\t{s=}')
                    # Load state_dict
                    state_dict_dir = list(filter(lambda f: f.startswith(f'{s}_'),
                                                 os.listdir(models_dir)))
                    if state_dict_dir:
                        state_dict = torch.load(
                            os.path.join(models_dir, state_dict_dir[0]),
                            weights_only=False)
                        
                        # Initialize model with balanced config
                        model = model_class(config)
                        model.load_state_dict(state_dict)
                        model.eval()
                        model.to(device)
                        
                        # Test
                        test_accs = test(device, model, test_loader, 1)
                        print(f'\t\t\t{test_accs}')
                        
                        p_accs.append(test_accs)
        
        p_accs = torch.Tensor(p_accs)
        p_accs = (p_accs.mean(), p_accs.std())
        print(f'\t\t\t{p_accs}')
        
        aux_dir = os.path.join(cwd, 'results', '6_privacy', model_class.name,
                               'attacks', 'acc_by_class', 'nn', str(p_indian))
        os.makedirs(aux_dir, exist_ok=True)
        
        torch.save(p_accs, os.path.join(aux_dir, f'{sex}_{indian}.pt'))


###############
# System args #
###############
# MARK: System args

if __name__ == '__main__':
    argv = sys.argv
    if len(argv) == 1:
        print('No argumets were passed')
        print('\t<sex>\n'
              '\t<indian>')
        sys.exit()
      
    # Read options and arguments
    try:
        opts, args = getopt.getopt(argv[1:], 'h', ['help'])
    except getopt.GetoptError:
        print('Available options are:\n'
              '\t--help, -h\n')
        sys.exit(2)

    sex = None
    indian = None
    if len(args) == 2:
        sex = int(args[0])
        indian = int(args[1])
    else:
        print('All arguments should be passed')
        print('\t<sex>\n'
              '\t<indian>')
        sys.exit()
        
    accs_by_class(sex, indian)
