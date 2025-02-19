# Guide to reproduce experiments:


1. Tune NN models:
    ```
    python experiments/0_train_nns/train.py --tune model_name <n_epochs> <n_samples> <p_english>
    ```

    - ``model_name = "fffc_tiny", "fffc_medium", "fffc_large"``
    - ``n_epochs = 5``
    - ``n_samples = 100``
    - ``p_english = 0.5``


2. Train NN models:
    ```
    python experiments/0_train_nns/train.py model_name <n_epochs> <n_samples> <p_english>
    ```

    - ``model_name = "fffc_tiny", "fffc_medium", "fffc_large"``
    - ``n_epochs = 20``
    - ``n_samples = 25``
    - ``p_english = 0.005, 0.01, 0.05,  0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.995``
