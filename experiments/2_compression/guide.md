# Guide to reproduce experiments:


1. Train one model of each type using ``experiments/0_train_nns/train.py``:
   ```
   python experiments/0_train_nns/train.py --test <model_name> <n_epochs> <p_english> <idx>
   ```

   - ``<model_name> = "fffc_tiny", "fffc_medium", "fffc_large"``
   - ``n_epochs = 100``
   - ``p_english = 0.5``
   - ``idx = 0``

2. Move trained models from ``models`` to ``results/2_compression``, and rename
   them as ``<model_name>_<n_epochs>_<p_english>_<idx>_<test_acc>.pt``


3. Compress models with TT-RSS for different bond dimensions, repeating
   compression 5 times:
   ```
   python experiments/2_compression/compression.py <model_name> <embedding_name> <embed_dim> <bond_dim> <domain_multiplier> <n_samples> <n_models>
   ```

   - ``<model_name> = "fffc_tiny", "fffc_medium", "fffc_large"``
   - ``embedding_name = "poly"``
   - ``embed_dim = 2``
   - ``bond_dim = 2, 5, 10``
   - ``domain_multiplier = 1``
   - ``n_samples = 100``
   - ``n_models = 5``


4. Take the best model from the 5 previously compressed ones and re-train for
   10 epochs on small unseen dataset:
   ```
   python experiments/2_compression/retrain_best.py <model_name> <bond_dim> <n_epochs>
   ```

   - ``<model_name> = "fffc_tiny", "fffc_medium", "fffc_large"``
   - ``bond_dim = 2, 5, 10``
   - ``n_epochs = 10``


5. Compress again each model via Layer-Wise tensorization with different bond
   dimensions, and retrain each model 10 epochs on small unseen dataset:
   ```
   python experiments/2_compression/compression_lw.py <model_name> <bond_dim> <n_epochs>
   ```

   - ``<model_name> = "fffc_tiny_tn", "fffc_medium_tn", "fffc_large_tn"``
   - ``bond_dim = 2, 5, 10``
   - ``n_epochs = 10``


* Results and figures are processed in ``compression.ipynb``.
