# Guide to reproduce experiments:


1. Train TT models with different embeddings and initializations:
    ```
    python experiments/5_initialization/train.py <init_method> <embedding> <renormalize> <bond_dim> <n_epochs>
    ```

    - ``init_method = "rss", "rss_random", "randn_eye", "unit", "canonical"``
    - ``embedding = "poly", "unit"``
    - ``renormalize = False``
    - ``bond_dim = 5``
    - ``n_epochs = 100``


* Results and figures are processed in ``trainings.ipynb``.
