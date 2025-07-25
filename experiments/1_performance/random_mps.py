import os
import sys
import getopt
import time

import torch
import tntorch as tn

import tensorkrowch as tk
from tensorkrowch.decompositions import tt_rss


torch.set_num_threads(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.double
eps = 1e-15

cwd = os.getcwd()


# Tensorize
# =========

def tt_rss_tensorization(n_features, phys_dim, bond_dim1, bond_dim2,
                         samples_size, sketch_size, verbose=False):
    cores_dir = os.path.join(cwd, 'results', '1_performance', 'random_mps')
    os.makedirs(cores_dir, exist_ok=True)
    
    results_dir = os.path.join(
        cwd, 'results', '1_performance', 'random_mps',
        f'rss_{n_features}_{phys_dim}_{bond_dim1}_{bond_dim2}_{samples_size}_{sketch_size}')
    os.makedirs(results_dir, exist_ok=True)
    
    cores_file = f'cores_{n_features}_{phys_dim}_{bond_dim1}.pt'
    if not os.path.exists(os.path.join(cores_dir, cores_file)):
        mps = tk.models.MPS(n_features=n_features,
                            phys_dim=phys_dim,
                            bond_dim=bond_dim1,
                            init_method='unit',
                            device=device,
                            dtype=dtype)
        
        torch.save(mps.tensors, os.path.join(cores_dir, cores_file))
    
    cores = torch.load(os.path.join(cores_dir, cores_file),
                       weights_only=False)
    
    mps = tk.models.MPS(tensors=cores)
    mps.trace(torch.zeros(1, n_features, phys_dim, device=device, dtype=dtype))

    domain = torch.arange(2)

    sketch_samples = torch.randint(low=0, high=phys_dim, size=(sketch_size, n_features))

    def embedding(x): return tk.embeddings.basis(x, dim=phys_dim).float()
    def fun(x): return mps(embedding(x).to(dtype)).unsqueeze(-1)
    
    if verbose: print('* Starting TT-RSS tensorization...')
    cores_rss, info = tt_rss(function=fun,
                             embedding=embedding,
                             sketch_samples=sketch_samples,
                             domain=domain,
                             rank=bond_dim2,
                             cum_percentage=1-1e-5,
                             batch_size=500,
                             device=device,
                             dtype=dtype,
                             verbose=verbose,
                             return_info=True)
    
    if verbose:
        print(f'* Tensorization finished in {info["total_time"]:.2e} seconds')
        print(f'--> Sketch error: {info["val_eps"]:.2e}')
    
    # Relative error
    mps_rss = tk.models.MPS(tensors=[c.to(device) for c in cores_rss])
    mps_rss.trace(torch.zeros(1, n_features, phys_dim, device=device, dtype=dtype))
    
    samples = torch.randint(low=0, high=phys_dim, size=(samples_size, n_features))
    exact_results = mps(embedding(samples.to(device)).to(dtype))
    rss_results = mps_rss(embedding(samples.to(device)).to(dtype))
    rel_error = (exact_results - rss_results).norm() / (exact_results.norm() + eps)
    
    if verbose: print(f'--> Relative error: {rel_error:.2e}')
    
    # Fidelity
    mps.reset()
    mps.unset_data_nodes()
    mps_rss.reset()
    mps_rss.unset_data_nodes()
    
    mps_norm = mps.norm(log_scale=True)
    mps_rss_norm = mps_rss.norm(log_scale=True)

    mps.reset()
    mps_rss.reset()
    
    for node1, node2 in zip(mps.mats_env, mps_rss.mats_env):
        node1['input'] ^ node2['input']
    
    log_scale = 0

    # Contract mps with mps_rss
    stack = tk.stack(mps.mats_env)
    stack_rss = tk.stack(mps_rss.mats_env)
    stack ^ stack_rss

    mats_results = tk.unbind(stack @ stack_rss)

    mats_results[0] = mps.left_node @ (mps_rss.left_node @ mats_results[0])
    mats_results[-1] = (mats_results[-1] @ mps.right_node) @ mps_rss.right_node

    result = mats_results[0]
    for mat in mats_results[1:]:
        result @= mat
        
        log_scale += result.norm().log()
        result = result.renormalize()

    approx_mps_norm = (result.tensor.log() + log_scale) / 2
    fidelity = (2*approx_mps_norm - mps_norm - mps_rss_norm).exp()
    
    if verbose: print(f'--> Fidelity: {fidelity:.2e}')
    
    torch.save(
        cores_rss,
        os.path.join(results_dir,
                     f'{info["total_time"]:.2e}_{info["val_eps"]:.2e}_'
                     f'{rel_error:.2e}_{fidelity:.2e}.pt'))


def tt_cross_tensorization(n_features, phys_dim, bond_dim1, bond_dim2,
                           samples_size, verbose=False):
    cores_dir = os.path.join(cwd, 'results', '1_performance', 'random_mps')
    os.makedirs(cores_dir, exist_ok=True)
    
    results_dir = os.path.join(
        cwd, 'results', '1_performance', 'random_mps',
        f'cross_{n_features}_{phys_dim}_{bond_dim1}_{bond_dim2}_{samples_size}')
    os.makedirs(results_dir, exist_ok=True)
    
    cores_file = f'cores_{n_features}_{phys_dim}_{bond_dim1}.pt'
    if not os.path.exists(os.path.join(cores_dir, cores_file)):
        mps = tk.models.MPS(n_features=n_features,
                            phys_dim=phys_dim,
                            bond_dim=bond_dim1,
                            init_method='unit',
                            device=device,
                            dtype=dtype)
        
        torch.save(mps.tensors, os.path.join(cores_dir, cores_file))
    
    cores = torch.load(os.path.join(cores_dir, cores_file),
                       weights_only=False)
    
    mps = tk.models.MPS(tensors=cores)
    mps.trace(torch.zeros(1, n_features, phys_dim, device=device, dtype=dtype))

    domain = [torch.arange(2, device=device, dtype=dtype)] * n_features

    def embedding(x): return tk.embeddings.basis(x.int(), dim=phys_dim).float()
    def fun(x): return mps(embedding(x).to(dtype))
    
    if verbose: print('* Starting TT-CI tensorization...')
    
    start = time.time()
    tt_cross, info = tn.cross(function=fun,
                              domain=domain,
                              device=device,
                              function_arg='matrix',
                              # rmax=bond_dim2,
                              ranks_tt=bond_dim2,
                              max_iter=1,   # 25 by default
                              eps=1e-15,    # 1e-6 by default
                              verbose=verbose,
                              return_info=True)
    all_time = time.time() - start
    
    if verbose:
        print(f'* Tensorization finished in {info["total_time"]:.2e} seconds')
        print(f' (Total time {all_time:.2e} seconds) ')
        print(f'--> Val. error: {info["val_eps"]:.2e}')
    
    # Relative error
    cores_cross = tt_cross.cores
    cores_cross[0] = cores_cross[0][0]
    cores_cross[-1] = cores_cross[-1][..., 0]
    
    mps_cross = tk.models.MPS(tensors=[c.to(device) for c in cores_cross])
    mps_cross.trace(torch.zeros(1, n_features, phys_dim, device=device, dtype=dtype))
    
    samples = torch.randint(low=0, high=phys_dim, size=(samples_size, n_features))
    exact_results = mps(embedding(samples.to(device)).to(dtype))
    cross_results = mps_cross(embedding(samples.to(device)).to(dtype))
    rel_error = (exact_results - cross_results).norm() / (exact_results.norm() + eps)
    
    if verbose: print(f'--> Relative error: {rel_error:.2e}')
    
    # Fidelity
    mps.reset()
    mps.unset_data_nodes()
    mps_cross.reset()
    mps_cross.unset_data_nodes()
    
    mps_norm = mps.norm(log_scale=True)
    mps_cross_norm = mps_cross.norm(log_scale=True)

    mps.reset()
    mps_cross.reset()
    
    for node1, node2 in zip(mps.mats_env, mps_cross.mats_env):
        node1['input'] ^ node2['input']
    
    log_scale = 0

    # Contract mps with mps_cross
    stack = tk.stack(mps.mats_env)
    stack_rss = tk.stack(mps_cross.mats_env)
    stack ^ stack_rss

    mats_results = tk.unbind(stack @ stack_rss)

    mats_results[0] = mps.left_node @ (mps_cross.left_node @ mats_results[0])
    mats_results[-1] = (mats_results[-1] @ mps.right_node) @ mps_cross.right_node

    result = mats_results[0]
    for mat in mats_results[1:]:
        result @= mat
        
        log_scale += result.norm().log()
        result = result.renormalize()

    approx_mps_norm = (result.tensor.log() + log_scale) / 2
    fidelity = (2*approx_mps_norm - mps_norm - mps_cross_norm).exp()
    
    if verbose: print(f'--> Fidelity: {fidelity:.2e}')
    
    torch.save(
        cores_cross,
        os.path.join(results_dir,
                     f'{all_time:.2e}_{info["total_time"]:.2e}_'
                     f'{rel_error:.2e}_{fidelity:.2e}.pt'))


# Tensorize multiple times
# ========================

def multiple_tt_rss(n, n_features, phys_dim, bond_dim1, bond_dim2,
                    samples_size, sketch_size):
    for _ in range(n):
        tt_rss_tensorization(n_features=n_features,
                             phys_dim=phys_dim,
                             bond_dim1=bond_dim1,
                             bond_dim2=bond_dim2,
                             samples_size=samples_size,
                             sketch_size=sketch_size)


def multiple_tt_cross(n, n_features, phys_dim, bond_dim1, bond_dim2, samples_size):
    for _ in range(n):
        tt_cross_tensorization(n_features=n_features,
                               phys_dim=phys_dim,
                               bond_dim1=bond_dim1,
                               bond_dim2=bond_dim2,
                               samples_size=samples_size)


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
              '\t--rss\n'
              '\t--cross\n'
              '\t--n')
        sys.exit()
      
    # Read options and arguments
    try:
        opts, args = getopt.getopt(argv[1:], 'h', ['help', 'rss', 'cross', 'n'])
    except getopt.GetoptError:
        print('Available options are:\n'
              '\t--help, -h\n'
              '\t--rss\n'
              '\t--cross\n'
              '\t--n')
        sys.exit(2)
    
    # Save selected options
    options = {'rss': False,
               'cross': False,
               'n': False}
    
    for opt, arg in opts:
        if (opt == '-h') or (opt == '--help'):
            print('Available options are:\n'
                  '\t--help, -h\n'
                  '\t--rss\n'
                  '\t--cross\n'
                  '\t--n')
            sys.exit()
        elif opt == '--rss':
            options['rss'] = True
        elif opt == '--cross':
            options['cross'] = True
        elif opt == '--n':
            options['n'] = True
    
    # Check if selected options are compatible
    if options['rss'] and options['cross']:
        print('Options "rss" and "cross" are incompatible')
        sys.exit()
    elif not (options['rss'] or options['cross']):
        print('One of the options "rss" and "cross" should be chosen')
        sys.exit()
    
    # Multiple
    if options['n']:
        # RSS
        if options['rss']:
            if len(args) < 7:
                print('In "n" and "rss" mode the following arguments need '
                      'to be passed:\n'
                      '\t1) n\n'
                      '\t2) n_features\n'
                      '\t3) phys_dim\n'
                      '\t4) bond_dim1\n'
                      '\t5) bond_dim2\n'
                      '\t6) samples_size\n'
                      '\t7) sketch_size')
                sys.exit()
            else:
                n = int(args[0])
                n_features = int(args[1])
                phys_dim = int(args[2])
                bond_dim1 = int(args[3])
                bond_dim2 = int(args[4])
                samples_size = int(args[5])
                sketch_size = int(args[6])
            
            multiple_tt_rss(n=n,
                            n_features=n_features,
                            phys_dim=phys_dim,
                            bond_dim1=bond_dim1,
                            bond_dim2=bond_dim2,
                            samples_size=samples_size,
                            sketch_size=sketch_size)
        
        # CROSS
        if options['cross']:
            if len(args) < 6:
                print('In "n" and "cross" mode the following arguments need '
                      'to be passed:\n'
                      '\t1) n\n'
                      '\t2) n_features\n'
                      '\t3) phys_dim\n'
                      '\t4) bond_dim1\n'
                      '\t5) bond_dim2\n'
                      '\t6) samples_size')
                sys.exit()
            else:
                n = int(args[0])
                n_features = int(args[1])
                phys_dim = int(args[2])
                bond_dim1 = int(args[3])
                bond_dim2 = int(args[4])
                samples_size = int(args[5])
            
            multiple_tt_cross(n=n,
                              n_features=n_features,
                              phys_dim=phys_dim,
                              bond_dim1=bond_dim1,
                              bond_dim2=bond_dim2,
                              samples_size=samples_size)
    
    else:
        # RSS
        if options['rss']:
            if len(args) < 5:
                print('In "rss" mode the following arguments need to be passed:\n'
                      '\t1) n_features\n'
                      '\t2) phys_dim\n'
                      '\t3) bond_dim\n'
                      '\t4) samples_size\n'
                      '\t5) sketch_size')
                sys.exit()
            else:
                n_features = int(args[0])
                phys_dim = int(args[1])
                bond_dim1 = int(args[3])
                bond_dim2 = int(args[4])
                samples_size = int(args[5])
                sketch_size = int(args[6])
            
            tt_rss_tensorization(n_features=n_features,
                                 phys_dim=phys_dim,
                                 bond_dim1=bond_dim1,
                                 bond_dim2=bond_dim2,
                                 samples_size=samples_size,
                                 sketch_size=sketch_size,
                                 verbose=True)
        
        # CROSS
        if options['cross']:
            if len(args) < 4:
                print('In "cross" mode the following arguments need to be passed:\n'
                      '\t1) n_features\n'
                      '\t2) phys_dim\n'
                      '\t3) bond_dim\n'
                      '\t4) samples_size')
                sys.exit()
            else:
                n_features = int(args[0])
                phys_dim = int(args[1])
                bond_dim1 = int(args[3])
                bond_dim2 = int(args[4])
                samples_size = int(args[5])
            
            tt_cross_tensorization(n_features=n_features,
                                   phys_dim=phys_dim,
                                   bond_dim1=bond_dim1,
                                   bond_dim2=bond_dim2,
                                   samples_size=samples_size,
                                   verbose=True)
