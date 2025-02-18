# Guide to reproduce experiments:

0) Download dataset [Common Voice Corpus 16.1](https://commonvoice.mozilla.org/en/datasets),
   and extract files in folder ``CommonVoice``. In this folder you should place
   the ``clips`` folder and all the .tsv files.

1) Create all dataset splits for train / val / test:
    python audio_create_datasets.py

2) Create all test datasets for privacy attacks:
    python audio_create_test_datasets.py
