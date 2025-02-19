# [Tensorization of neural networks for improved privacy and interpretability](https://arxiv.org/abs/2501.06300)

**José Ramón Pareja Monturiol, Alejandro Pozas-Kerstjens, David Pérez-García**


This repository contains all the code used to run the experiments in the paper,
including the tuning and training of neural network models, tensorizing them,
and training tensor train models.  

The code for the TT-RSS decomposition is not included in this repository, as it
is already available in [TensorKrowch](https://github.com/joserapa98/tensorkrowch).  


## Requirements

* python >= 3.8

**Tensor network models**
* tensorkrowch == 1.1.6

Install via:
```
pip install tensorkrowch==1.1.6
```

**Deep learning framework** (versions specified by TensorKrowch)
* torch
* torchvision  
* torchaudio  

**Hyperparameter tuning**
* ray >= 2.37.0

Install via:
```
pip install -U "ray[data,train,tune,serve]"
```

**Packages for figures**
* matplotlib
* seaborn

To use $\LaTeX$ in figure texts, make sure it is installed on your system.


## Instructions

To run the experiments, follow the instructions in each `guide` file, starting
with the one in the parent folder, which explains how to create the necessary
datasets.  

The code for the different experiments is located in the corresponding subfolders
inside the `experiments` directory. Each subfolder contains a `guide` file with
instructions on how to run the code. The parameters are preconfigured to reproduce
the results presented in the paper.  

Experiments are intended to be run sequentially, from `0_train_nns` to `6_privacy`.
However, if you are only interested in a specific experiment, you may not need
to run all of them. In that case, ensure that any required prior experiments
have been completed.  

All scripts should be executed from the parent folder.  


## Citing

If you would like to cite this work, please use the following format:

- J. R. Pareja Monturiol, A. Pozas-Kerstjens, D. Pérez-García, "Tensorization
of neural networks for improved privacy and interpretability" (2025), arXiv:2501.06300

```
@misc{pareja2025tensorization,
      title={Tensorization of neural networks for improved privacy and interpretability}, 
      author={Pareja Monturiol, José Ramón and Pozas-Kerstjens, Alejandro and Pérez-García, David},
      year={2025},
      eprint={2501.06300},
      archivePrefix={arXiv},
      primaryClass={math.NA},
      url={https://arxiv.org/abs/2501.06300}, 
}
```

