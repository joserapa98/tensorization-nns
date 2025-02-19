# Guide to reproduce experiments:


1. Tensorize, retrain and obfuscate all ``fffc_tiny`` models trained
   in ``0_train_nns``:
   ```
   python experiments/6_privacy/tensorize.py model_name <n_epochs> <p_english>
   ```

   - ``model_name = "fffc_tiny"``
   - ``n_epochs = 10``
   - ``p_english = 0.005, 0.01, 0.05,  0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.995``

   All models are tensorized with parameters:

   - ``input_dim = 2``
   - ``bond_dim = 5``
   - ``sketch_size = 100``


2. Compute mean accuracies of the different models in the different subclasses
   in the dataset:
   ```
   python experiments/6_privacy/plot_attack_bb_all_classes.py <model_type> <retrained> <private>
   ```

   Use the following combinations:

   - ``(model_type = "nn", retrained = False, private = False)``
   - ``(model_type = "mps", retrained = True, private = False)``
   - ``(model_type = "mps", retrained = True, private = True)``


3. Create Black-Box and White-Box representations of all the NN and MPS models:
   ```
   python experiments/6_privacy/attacks.py <model_type> <retrained> <private>
   ```

   Use the following combinations:

   - ``(model_type = "nn", retrained = False, private = False)``
   - ``(model_type = "mps", retrained = True, private = False)``
   - ``(model_type = "mps", retrained = True, private = True)``


* Results and figures are processed in ``attacks.ipynb``.
