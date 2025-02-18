"""
Re-train tensorized MPS models
"""

import os
import sys
import getopt
import copy
import json
from importlib import util

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision
import torchaudio

import tensorkrowch as tk
from tensorkrowch.utils import random_unitary

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
                                       batch_size=batch_size,
                                       collate_fn=none_collate,
                                       shuffle=False)
    test_unused_loader = DataLoader(test_unused_dataset,
                                    batch_size=batch_size,
                                    collate_fn=none_collate,
                                    shuffle=False)
    
    return test_tensorize_loader, test_unused_loader


def training_epoch_tn(device, model, embedding, criterion, optimizer,
                      train_loader, logs, n_batches=None):
    if n_batches is not None:
        n_batches = min(n_batches, len(train_loader))
    else:
        n_batches = len(train_loader)
    i = 0
    
    running_loss = 0
    running_acc = 0

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
        scores = torch.where(scores == 0, 1e-10, scores)
        scores = scores.log()
        
        loss = criterion(scores, labels)
        
        with torch.no_grad():
            _, preds = torch.max(scores, 1)
            accuracy = (preds == labels).float().mean().item()
            running_acc += accuracy
            running_loss += loss.item()
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient descent
        optimizer.step()
        
        i += 1
        if i >= n_batches:
            break
    
    logs['train_losses'].append(running_loss / n_batches)
    logs['train_accs'].append(running_acc / n_batches)
    
    return model, logs


def test_tn(device, model, embedding, criterion, test_loader,
            logs, n_batches=None):
    """Computes accuracy on test set."""
    running_loss = 0
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
                           inline_mats=False,
                           renormalize=True)
            scores = scores.pow(2)
            scores = scores / scores.norm(dim=1, keepdim=True)
            scores = torch.where(scores == 0, 1e-10, scores)
            scores = scores.log()
            
            loss = criterion(scores, labels)
            
            _, preds = torch.max(scores, 1)
            accuracy = (preds == labels).float().mean().item()
            running_acc += accuracy
            running_loss += loss.item()
            
            i += 1
            if i >= n_batches:
                break
    
    logs['val_losses'].append(running_loss / n_batches)
    logs['val_accs'].append(running_acc / n_batches)
    
    return logs


###############################################################################
###############################################################################

#############
# Tensorize #
#############
# MARK: Tensorize

def retrain_tn(model_class, p_indian, start_idx, end_idx):
    """Tensorizes models trained with the given ``p_indian`` and ``idx``."""
    global p_indian_list
    
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    
    if p_indian is not None:
        p_indian_list_aux = [p_indian]
    else:
        p_indian_list_aux = p_indian_list
        
    if start_idx is not None:
        idx_range_aux = range(start_idx, end_idx)
    else:
        idx_range_aux = range(10)
        
    for p_indian in p_indian_list_aux:
        for idx in idx_range_aux:
            cores_dir = os.path.join(cwd, 'results', '6_privacy',
                                     model_class.name, 'cores',
                                     str(p_indian), str(idx))
            sketches_dir = os.path.join(cwd, 'results', '6_privacy',
                                        model_class.name, 'sketches',
                                        str(p_indian), str(idx))
            
            recores_dir = os.path.join(cwd, 'results', '6_privacy',
                                       model_class.name, 'recores',
                                       str(p_indian), str(idx))
            os.makedirs(recores_dir, exist_ok=True)
            
            priv_recores_dir = os.path.join(cwd, 'results', '6_privacy',
                                            model_class.name, 'priv_recores',
                                            str(p_indian), str(idx))
            os.makedirs(priv_recores_dir, exist_ok=True)
            
            n_cores = len(os.listdir(recores_dir))
            n_recores = len(os.listdir(priv_recores_dir))
            
            for s in range(n_recores, n_cores):
                # Load cores
                aux_cores_dir = list(filter(lambda f: f.startswith(f'{s}_'),
                                             os.listdir(recores_dir)))[0]
                cores = torch.load(os.path.join(recores_dir, aux_cores_dir),
                                   weights_only=False)
                prev_test_acc = float(aux_cores_dir.split('_')[1])
                
                # Load data
                batch_size = 64
                tensorize_loader, unused_loader = load_sketch_samples(
                    p_indian, idx, batch_size)
                
                # Load sketch samples
                sketch_samples, sketch_labels = torch.load(
                    os.path.join(sketches_dir, f'{s}.pt'),
                    weights_only=False
                    )
                
                n_samples = 100
                n_features = out_rate // 2 + 1
                embed_dim = 2
                
                def embedding(x):
                    x = tk.embeddings.poly(x, degree=embed_dim - 1)
                    return x
                
                # MPS model
                tn_model = tk.models.MPSLayer(tensors=cores)
                tn_model.eval()
                tn_model.to(device)
                
                # for i, node in enumerate(tn_model.mats_env):
                #     U = random_unitary(node.size('right')).to(device)
                #     # U = torch.randint(low=0, high=2, size=(node.size('right'),))
                #     # U = (2 * U - 1).float().to(device)
                #     # U = torch.diag(U)
                    
                #     if i < (len(tn_model.mats_env) - 1):
                #         node.tensor = torch.einsum('lir,rk->lik', node.tensor, U)
                    
                #     if i > 0:
                #         node.tensor = torch.einsum('kl,lir->kir', prev_U, node.tensor)
                    
                #     prev_U = U.clone().H
                #     # prev_U = torch.linalg.inv(U.clone())
                #     # prev_U = U.clone()
                    
                #     # print(U @ prev_U)
                
                # Canonicalize
                tn_model.canonicalize(mode='svd', renormalize=True)
                
                # Fidelity
                old_tn_model = tk.models.MPSLayer(tensors=cores)
                old_tn_model.eval()
                old_tn_model.to(device)
                
                mps_norm = tn_model.norm(log_scale=True)
                mps2_norm = old_tn_model.norm(log_scale=True)
                
                print(mps_norm)
                print(mps2_norm)

                tn_model.reset()
                old_tn_model.reset()

                for node1, node2 in zip(tn_model.mats_env, old_tn_model.mats_env):
                    # print((node1.tensor - node2.tensor).norm())
                    # print(node1.tensor.norm() - node2.tensor.norm())
                    node1['input'] ^ node2['input']
                
                stack = tk.stack(tn_model.mats_env)
                stack_rss = tk.stack(old_tn_model.mats_env)
                stack ^ stack_rss

                mats_results = tk.unbind(stack @ stack_rss)

                mats_results[0] = tn_model.left_node @ (old_tn_model.left_node @ mats_results[0])
                mats_results[-1] = (mats_results[-1] @ tn_model.right_node) @ old_tn_model.right_node
                
                log_scale = 0
                
                result = mats_results[0]
                for mat in mats_results[1:]:
                    result @= mat
                    
                    log_scale += result.norm().log()
                    result = result.renormalize()

                approx_mps_norm = (result.tensor.log() + log_scale) / 2
                print(approx_mps_norm)
                print((2*approx_mps_norm - mps_norm - mps2_norm).exp())
                
                
                # MPS model
                tn_model = tk.models.MPSLayer(tensors=cores)
                tn_model.eval()
                tn_model.to(device)
                
                # for i, node in enumerate(tn_model.mats_env):
                #     U = random_unitary(node.size('right')).to(device)
                #     # U = torch.randint(low=0, high=2, size=(node.size('right'),))
                #     # U = (2 * U - 1).float().to(device)
                #     # U = torch.diag(U)
                    
                #     if i < (len(tn_model.mats_env) - 1):
                #         node.tensor = torch.einsum('lir,rk->lik', node.tensor, U)
                    
                #     if i > 0:
                #         node.tensor = torch.einsum('kl,lir->kir', prev_U, node.tensor)
                    
                #     prev_U = U.clone().H
                #     # prev_U = torch.linalg.inv(U.clone())
                #     # prev_U = U.clone()
                    
                #     # print(U @ prev_U)
                
                
                # Canonicalize
                tn_model.canonicalize(mode='svd', renormalize=True)
                
                # for t in tn_model.tensors:
                #     print(t.norm())
                # print()
                
                old_tn_model = tk.models.MPSLayer(tensors=cores)
                old_tn_model.eval()
                old_tn_model.to(device)
                
                log_norm = 0

                for node in old_tn_model.mats_env:
                    log_norm += node.norm().log()
                    node.tensor = node.tensor / node.norm()

                for node in old_tn_model.mats_env:
                    node.tensor = node.tensor * (log_norm / len(old_tn_model.mats_env)).exp()
                
                # for node in old_tn_model.mats_env:
                #     print(node.norm())
                
                # for t in old_tn_model.tensors:
                #     print(t.norm())
                # print()
                
                
                tn_model.trace(
                    torch.zeros(1, n_features - 1, embed_dim).to(device),
                    inline_input=False,
                    inline_mats=False,
                    renormalize=False
                )
                
                old_tn_model.trace(
                    torch.zeros(1, n_features - 1, embed_dim).to(device),
                    inline_input=False,
                    inline_mats=False,
                    renormalize=False
                )
                
                scores = tn_model(embedding(sketch_samples[:10].to(device)),
                           inline_input=False,
                           inline_mats=False,
                           renormalize=False)
                print(scores)

                scores = old_tn_model(embedding(sketch_samples[:10].to(device)),
                           inline_input=False,
                           inline_mats=False,
                           renormalize=False)
                print(scores)
                
                
                old_tn_model = tk.models.MPSLayer(tensors=cores)
                old_tn_model.eval()
                old_tn_model.to(device)
                
                old_tn_model.trace(
                    torch.zeros(1, n_features - 1, embed_dim).to(device),
                    inline_input=False,
                    inline_mats=False,
                    renormalize=False
                )

                scores = old_tn_model(embedding(sketch_samples[:10].to(device)),
                           inline_input=False,
                           inline_mats=False,
                           renormalize=False)
                print(scores)
                
                
                
                # tn_model = tk.models.MPSLayer(tensors=cores)
                # tn_model.eval()
                # tn_model.to(device)
                
                # log_norm = 0
                # norms = []
                # for node in tn_model.mats_env:
                #     norms.append(node.norm().detach())
                #     node.tensor = node.tensor / node.norm()
                
                # tn_model.canonicalize(mode='svdr', renormalize=True)

                # for i, node in enumerate(tn_model.mats_env):
                #     node.tensor = node.tensor * norms[i]
                
                # tn_model.trace(
                #     torch.zeros(1, n_features - 1, embed_dim).to(device),
                #     inline_input=False,
                #     inline_mats=False,
                #     renormalize=False
                # )
                
                # scores = tn_model(embedding(sketch_samples[:10].to(device)),
                #            inline_input=False,
                #            inline_mats=False,
                #            renormalize=False)
                # print(scores)
                

                
                criterion = nn.NLLLoss()
                
                logs = {'train_losses': [],
                        'val_losses': [],
                        'train_accs': [],
                        'val_accs': []}
                
                logs = test_tn(device=device,
                               model=tn_model,
                               embedding=embedding,
                               criterion=criterion,
                               test_loader=unused_loader,
                               logs=logs,
                               # n_batches=10,
                               )
                
                test_acc = logs['val_accs'][-1]
                
                # Test on sketch samples
                sketch_dataset = TensorDataset(sketch_samples[:n_samples],
                                               sketch_labels[:n_samples])
                sketch_loader = DataLoader(sketch_dataset,
                                           batch_size=batch_size,
                                           collate_fn=none_collate,
                                           shuffle=False)
                
                logs = test_tn(device=device,
                               model=tn_model,
                               embedding=embedding,
                               criterion=criterion,
                               test_loader=sketch_loader,
                               logs=logs,
                               # n_batches=10,
                               )
                
                sketch_acc = logs['val_accs'][-1]
                
                print(f'**{model_class.name}** (p: {p_indian}, i: {idx}, '
                      f's: {s}) => '
                      f'Prev. Test Acc.: {prev_test_acc:.4f}, '
                      f'Test Acc.: {test_acc:.4f}, '
                      f'Sketch Acc.: {sketch_acc:.4f}')
                
                

                logs = test_tn(device=device,
                               model=old_tn_model,
                               embedding=embedding,
                               criterion=criterion,
                               test_loader=unused_loader,
                               logs=logs,
                               # n_batches=10,
                               )
                
                test_acc = logs['val_accs'][-1]
                
                logs = test_tn(device=device,
                               model=old_tn_model,
                               embedding=embedding,
                               criterion=criterion,
                               test_loader=sketch_loader,
                               logs=logs,
                               # n_batches=10,
                               )
                
                sketch_acc = logs['val_accs'][-1]
                
                print(f'**{model_class.name}** (p: {p_indian}, i: {idx}, '
                      f's: {s}) => '
                      f'Prev. Test Acc.: {prev_test_acc:.4f}, '
                      f'Test Acc.: {test_acc:.4f}, '
                      f'Sketch Acc.: {sketch_acc:.4f}')
                
                # torch.save(
                #     tn_model.tensors,
                #     os.path.join(priv_recores_dir,
                #                  f'{s}_{test_acc:.4f}_{sketch_acc:.4f}.pt')
                # )


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
              '\t<n epochs>\n'
              f'\t(optional) proportion of imbalance (p in {p_indian_list})\n'
              '\t(optional) index of dataset (0 <= i <= 9)')
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
    start_idx = None
    end_idx = None
    if len(args) == 1:
        model_name = args[0]
    elif len(args) == 2:
        model_name = args[0]
        if args[1].isdigit():
            start_idx = int(args[1])
            end_idx = int(args[1]) + 1
        else:
            p_indian = float(args[1])
    elif len(args) == 3:
        model_name = args[0]
        p_indian = float(args[1])
        start_idx = int(args[2])
        end_idx = int(args[2]) + 1
    elif len(args) == 4:
        model_name = args[0]
        p_indian = float(args[1])
        start_idx = int(args[2])
        end_idx = int(args[3])
        
    aux_mod = import_file('model', os.path.join(cwd, 'models', f'{model_name}.py'))
    model_class = aux_mod.Model
    retrain_tn(model_class=model_class,
               p_indian=p_indian,
               start_idx=start_idx,
               end_idx=end_idx)
