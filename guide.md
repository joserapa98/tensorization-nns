# Guide to reproduce experiments:


1. Download dataset [Common Voice Corpus 16.1](https://commonvoice.mozilla.org/en/datasets),
   and extract files in folder ``CommonVoice``. In this folder you should place
   the ``clips`` folder and all the ``.tsv`` files.


2. Create all dataset splits for train / val / test:
    ```
    python audio_create_datasets.py
    ```

3. Create all test datasets for privacy attacks:
    ```
    python audio_create_test_datasets.py
    ```
