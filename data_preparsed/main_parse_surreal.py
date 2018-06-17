# This file preparsed surreal data into binary files
# For fast tfrecord read-write
from load_surreal import get_training_params
import os
import numpy as np
import pickle as pkl
from write_utils import write_syn_to_bin, read_syn_to_bin

def check_dir(path):
  if not os.path.exists(path):
    os.mkdir(path)

data_path = "/home/local/data/surreal/SURREAL/data/cmu/"
output_path = "/home/local/data/surreal/SURREAL/data/bin/"
is_test = False
modalities = ['train', 'test', 'val']
runs = ['run0', 'run1', 'run2']

for modality in modalities:
    for run in runs:
        print(modality, run)
        data_folder = os.path.join(data_path, modality, run)
        output_folder = os.path.join(output_path, modality, run)
        if not os.path.exists(data_folder):
            continue
        check_dir(output_folder)
        for filename in os.listdir(data_folder):
            file_folder = os.path.join(data_folder, filename)
            subfiles = [fname[:-4] for fname in os.listdir(file_folder) if fname.endswith('.mp4')]
            for sfile in subfiles:
                print sfile
                parsed_data = get_training_params(sfile, data_dir = data_folder)
                write_syn_to_bin(parsed_data, os.path.join(output_folder, sfile) + ".bin")
