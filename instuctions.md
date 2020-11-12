# Instructions for running meta-learning experiments

1. Run `pip install -e .` to install this package

2. Run `pip install -r requirements.txt`

3. Go to our shared google drive folder to download vocab files. The name of vocab files
indicates the corresponding dataset. `vocab_7ds_5w.txt` means that this is a meta vocab that contains
the top 50000 frequent words of all the 7 supported datasets. You can just download this file
for meta experiments

4. Go to `pointer_generator/utils/config.py` and update all the configs at the bottom of the file 
under `###New configs###`. `dataset_cache_dir` and `vocab_cache_dir` are places to store datasets and vocab.
`meta_vocab_file` is the name of the vocab file that you're using (downloaded from google drive).
The `meta_vocab_file` is assumed to locate under `vocab_cache_dir`. Update `meta_vocab_size` if you use a vocab
with size larger than 50000, like `vocab_7ds_7w.txt`.

5. Run `meta_train.py`, `meta_test.py`, and `meta_eval.py`