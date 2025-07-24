# Guide to reproduce experiments:


1. **TT-RSS:** Use ``random_mps.py``, ``slater_functions.py``, ``bars_stripes.py``
   and ``mnist.py`` to tensorize these functions with different configurations
   of hyperparameters:
    

   1. **Random TT:**
      ```
      python experiments/1_performance/random_mps.py --rss --n <n> <n_features> <phys_dim> <bond_dim1> <bond_dim2> <samples_size> <sketch_size>
      ```

      - ``n = 10``
      - ``n_features = 100, 200, 500``
      - ``phys_dim = 2``
      - ``bond_dim1 = 10``
      - ``bond_dim2 = sketch_size``
      - ``samples_size = 1000``
      - ``sketch_size = 10, 15, 20, 25, 30, 35``
    

   2. **Slater functions:**
      ```
      python experiments/1_performance/slater_functions.py --rss --n <n> <L> <m> <l> <bond_dim> <samples_size> <sketch_size>
      ```

      - ``n = 10``
      - ``L = 10``
      - ``m = 5``
      - ``l = 20, 40, 100``
      - ``bond_dim = 10``
      - ``samples_size = 1000``
      - ``sketch_size = 30, 40, 50, 60, 70, 80, 90, 100``
    

   3. **Bars and Stripes:**
      ```
      python experiments/1_performance/bars_stripes.py --rss --n <n> <n_features> <phys_dim> <bond_dim> <samples_size> <sketch_size>
      ```

      - ``n = 10``
      - ``n_features = 12, 16, 20``
      - ``phys_dim = 2``
      - ``bond_dim = 10``
      - ``samples_size = 1000``
      - ``sketch_size = 30, 40, 50, 60, 70, 80, 90, 100``
    

   4. **MNIST:**
      ```
      python experiments/1_performance/mnist.py --rss --n <n> <n_features> <phys_dim> <bond_dim> <samples_size> <sketch_size>
      ```

      - ``n = 10``
      - ``n_features = 12, 16, 20``
      - ``phys_dim = 2``
      - ``bond_dim = 10``
      - ``samples_size = 1000``
      - ``sketch_size = 30, 40, 50, 60, 70, 80, 90, 100``


1. **TT-CI:** Use ``random_mps.py``, ``slater_functions.py``, ``bars_stripes.py``
   and ``mnist.py`` to tensorize these functions with different configurations
   of hyperparameters:
    

   1. **Random TT:**
      ```
      python experiments/1_performance/random_mps.py --cross --n <n> <n_features> <phys_dim> <bond_dim1> <bond_dim2> <samples_size>
      ```

      - ``n = 10``
      - ``n_features = 100, 200 (500)``
      - ``phys_dim = 2``
      - ``bond_dim1 = 10``
      - ``bond_dim2 = 10``
      - ``samples_size = 1000``
    

   2. **Slater functions:**
      ```
      python experiments/1_performance/slater_functions.py --cross --n <n> <L> <m> <l> <bond_dim> <samples_size>
      ```

      - ``n = 10``
      - ``L = 10``
      - ``m = 5``
      - ``l = 20, 40, 100``
      - ``bond_dim = 10``
      - ``samples_size = 1000``
    

   3. **Bars and Stripes:**
      ```
      python experiments/1_performance/bars_stripes.py --cross --n <n> <n_features> <phys_dim> <bond_dim> <samples_size>
      ```

      - ``n = 10``
      - ``n_features = 12, 16, 20``
      - ``phys_dim = 2``
      - ``bond_dim = 10``
      - ``samples_size = 1000``
    

   4. **MNIST:**
      ```
      python experiments/1_performance/mnist.py --cross --n <n> <n_features> <phys_dim> <bond_dim> <samples_size>
      ```

      - ``n = 10``
      - ``n_features = 12 (16, 20)``
      - ``phys_dim = 2``
      - ``bond_dim = 10``
      - ``samples_size = 1000``


* Results and figures are processed in ``performance.ipynb``.
