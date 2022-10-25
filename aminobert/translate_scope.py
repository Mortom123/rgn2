import sys
import os
import time
import copy
import subprocess
import shutil
import pickle
import random
import glob

from Bio import SeqIO
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append('../../')
import modeling
import tokenization
import optimization
import run_finetuning_and_prediction


def run_prediction(seqs, qfunc, checkpoint_file, wt_log_prob_mat=None,
                   return_seq_log_probs=True, return_seq_output=True,
                   clip_seq_level_outputs=True):
    start = time.time()

    MAX_SEQ_LENGTH = 1024
    output_dir = '../../data/test/'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    tokenizer = tokenization.FullTokenizer(k=1, token_to_replace_with_mask='X')

    result = run_finetuning_and_prediction.run_model(
        input_seqs=list(seqs),
        labels=qfunc,
        max_seq_length=MAX_SEQ_LENGTH,
        tokenizer=tokenizer,
        bert_config_file='AminoBERT_config_v2.json',
        output_dir=output_dir,
        init_checkpoint=checkpoint_file,
        do_training=False,  # No fine-tuning
        do_evaluation=False,
        do_prediction=True,  # Prediction only.
        num_train_epochs=3,
        learning_rate=5e-5,
        warmup_proportion=0.1,
        train_batch_size=16,
        eval_batch_size=32,
        predict_batch_size=32,
        use_tpu=False,
        return_seq_log_probs=return_seq_log_probs,
        return_seq_output=return_seq_output,  # encoder_layers[-1]
        encoding_layer_for_seq_rep=[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
        wt_log_prob_mat=wt_log_prob_mat,
        clip_seq_level_outputs=clip_seq_level_outputs
    )

    end = time.time()
    result['compute_time'] = end - start

    return result

PREPEND_M = True
DATA_DIR = 'round_6/'

CHECKPOINT = os.path.join('checkpoint',
        'AminoBERT_runs_v2_uniparc_dataset_v2_5-1024_fresh_start_model.ckpt-1100000')


def fasta_read(fasta_file):
    headers = []
    seqs = []
    for seq_record in SeqIO.parse(fasta_file, 'fasta'):
        headers.append(seq_record.id)
        seqs.append(str(seq_record.seq))

    return headers, seqs


# Sequences to predict structures for. 1 sequence per fasta.
sequence_path = "scope.fasta"
# Read in sequences.
headers, seqs = fasta_read(sequence_path)

# Add a stop char to each sequence to be consistent
# with how the model was trained.
headers = [h for h in headers]
seqs = [s + '*' for s in seqs]

print(len(seqs), len(headers))

# Prepend an M. Again reflective of how the model
# was trained.
if PREPEND_M:
    for i in range(len(seqs)):
        if seqs[i][0] != 'M':
            seqs[i] = 'M' + seqs[i]

# Remove sequences that are too long for the model
mask = np.array([len(s) for s in seqs]) <= 1023
print('Sequences being removed due to length:', np.sum(~mask))
print('Sequences being removed:', np.array(headers)[~mask], np.array(seqs)[~mask])

seqs = list(np.array(seqs)[mask])
headers = list(np.array(headers)[mask])


def batched(iterable, n=1):
    current_batch = []
    for item in iterable:
        current_batch.append(item)
        if len(current_batch) == n:
            yield current_batch
            current_batch = []
    if current_batch:
        yield current_batch


for bheaders, bseqs in batched(zip(headers, seqs), n=300):
    qfunc = np.random.randn(len(bseqs))  # dummy labels. Ignore this.
    inf_result = run_prediction(bseqs, qfunc, CHECKPOINT)

    embedding_dir = "/beegfs/.global1/ws/s0794732-aminobert/embeddings"
    print('Writing numpy arrays to', embedding_dir)
    for j in range(len(bseqs)):
        embedding = inf_result['predict']['seq_output'][j]
        header = bheaders[j]
        assert embedding.shape[0] == len(bseqs[j])

        outfile = os.path.join(embedding_dir, header + '.npy')
        np.save(outfile, embedding)