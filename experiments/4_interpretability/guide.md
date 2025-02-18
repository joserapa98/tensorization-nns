# Guide to reproduce experiments:


1. Tensorize AKLT state:
    ```
    python experiments/4_interpretability/aklt.py --n <n> <n_features> <samples_size> <sketch_size>
    ```

    - ``n = 10``
    - ``n_features = 100, 200, 500``
    - ``samples_size = 1000``
    - ``sketch_size = 2, 4, 6, 8, 10, 12, 14``


* Results and figures are processed in ``aklt_figure.ipynb``.
