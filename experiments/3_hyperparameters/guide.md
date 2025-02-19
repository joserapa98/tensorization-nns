# Guide to reproduce experiments:


1. After completing **step 2** from ``2_compression``, copy the model
   corresponding to ``fffc_tiny`` to ``results/3_hyperparameters``.


2. Compress previous model several times for different configurations of
   hyperparameters:
   ```
   python experiments/2_compression/compression_hyperparameters.py model_name <embedding_name> <embed_dim> <bond_dim> <domain_multiplier> <n_samples> <n_models>
   ```

   - ``model_name = "fffc_tiny"``
   - ``embedding_name = "poly"``
   - ``embed_dim = 2, 3, 4, 5, 6``
   - ``bond_dim = 2, 5, 10, 20, 50``
   - ``domain_multiplier = 1``
   - ``n_samples = 10, 20, 50, 100, 200``
   - ``n_models = 50``

   Instead of computing all combinations, we start from the baseline point
   ``(embed_dim = 2, bond_dim = 5, n_samples = 50)``, and only move one these
   variables at a atime.


* Results and figures are processed in ``hyperparameters.ipynb``.
