# -*- coding: utf-8 -*-

import os

SENTENCE_STA = '<s>'
SENTENCE_END = '</s>'

UNK = 0
PAD = 1
BOS = 2
EOS = 3

PAD_TOKEN = '[PAD]'
UNK_TOKEN = '[UNK]'
BOS_TOKEN = '[BOS]'
EOS_TOKEN = '[EOS]'

beam_size=4
emb_dim= 128
batch_size= 4
hidden_dim= 256
max_enc_steps=400
max_dec_steps=100
max_tes_steps=100
min_dec_steps=35
vocab_size=50000

lr=0.15
cov_loss_wt = 1.0
pointer_gen = True
is_coverage = False

max_grad_norm=2.0
adagrad_init_acc=0.1
rand_unif_init_mag=0.02
trunc_norm_init_std=1e-4

eps = 1e-12
use_gpu=True
lr_coverage=0.15
max_iterations = 500000

# transformer
d_k = 64
d_v = 64
n_head = 6
tran = True
dropout = 0.1
n_layers = 6
d_model = 128
d_inner = 512
n_warmup_steps = 4000

root_dir = "/media/garage/data/pg" #os.path.expanduser("./")
log_root = os.path.join(root_dir, "dataset/log/")
#log_root = os.path.join(root_dir, "/scr-ssd/yanjunc/course/cs330/log")

#train_data_path = os.path.join(root_dir, "pointer_generator/dataset/finished_files/train.bin")

train_data_path = os.path.join(root_dir, "dataset/finished_files/chunked/train_*")
eval_data_path = os.path.join(root_dir, "dataset/finished_files/chunked/val_*")
decode_data_path = os.path.join(root_dir, "dataset/finished_files/chunked/test_*")
vocab_path = os.path.join(root_dir, "dataset/finished_files/vocab")

#train_data_path = os.path.join(root_dir, "/scr-ssd/yanjunc/course/cs330/dailymail_final/finished_files/chunked/train_*")
#eval_data_path = os.path.join(root_dir, "/scr-ssd/yanjunc/course/cs330/dailymail_final/finished_files/chunked/val_*")
#decode_data_path = os.path.join(root_dir, "/scr-ssd/yanjunc/course/cs330/dailymail_final/finished_files/chunked/test_*")
#vocab_path = os.path.join(root_dir, "/scr-ssd/yanjunc/course/cs330/dailymail_final/finished_files/vocab")


########################## New configs to fill ##############################
dataset_cache_dir = "/scr-ssd/yanjunc/course/cs330/huggingface/datasets"
vocab_cache_dir = "/scr-ssd/yanjunc/course/cs330/huggingface/vocab"

meta_train_datasets = "all"  # subset of HuggingFaceDataset.name_to_HFDS.keys()
meta_train_K = 8  # number of examples per task (dataset)
meta_test_datasets = "all"
meta_test_K = beam_size
meta_val_datasets = "all"
meta_val_K = 8
meta_vocab_file = "vocab_7ds_5w.txt"  # assume that this is in @vocab_cache_dir
meta_vocab_size = 50000
num_inner_loops = 2