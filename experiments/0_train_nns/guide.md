# Guide to reproduce experiments:

0. Install ray for hyperparameter tuning:
    ```
    pip install -U "ray[data,train,tune,serve]"
    ```

1. Tune models:
    ```
    python experiments/0_train_nns/train.py --tune <model_name> <n_epochs>(5) <n_samples>(100) <p_english>(0.5)
    ```

    * For ``<model_name>`` choose: 

2. Train models:
    ```
    python experiments/0_train_nns/train.py <model_name> <n_epochs>(20) <n_samples>(25) <p_english>
    ```
