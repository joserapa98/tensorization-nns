# Guide to reproduce experiments:


1. Download the [Common Voice Corpus 16.1](https://commonvoice.mozilla.org/en/datasets)
dataset and extract the files into a folder named ``CommonVoice`` inside the
parent directory. In this folder, place the ``clips`` folder and all the ``.tsv`` files.


2. Create all dataset splits for train / val / test:
    ```
    python audio_create_datasets.py
    ```

3. Create all test datasets for privacy attacks:
    ```
    python audio_create_test_datasets.py
    ```
