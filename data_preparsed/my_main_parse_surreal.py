# This file preparsed surreal data into binary files
# For fast tfrecord read-write
from load_surreal import get_training_params
import os
import numpy as np
import pickle as pkl
from write_utils import write_syn_to_bin, read_syn_to_bin

data_root_dir = '/home/local/mocap/data'
write_root_dir = '/home/local/mocap/output'
dataset_name = 'surreal'

# actions to parse
# actors to parse for each action
subjects = [10] #, 5, 6, 7, 8, 9, 11]
subactions = ['04']

def check_dir(path):
  if not os.path.exists(path):
    os.mkdir(path)

full_data_path = os.path.join(data_root_dir, dataset_name)
full_write_path = os.path.join(write_root_dir, dataset_name)

for subject in subjects:
    for subact in subactions:
        filename = str(subject) + "_" + subact

        if os.path.exists(os.path.join(full_data_path, filename)):
            data_folder = os.path.join(full_data_path, filename)
            output_folder = os.path.join(full_write_path, filename)
            check_dir(output_folder)

            subfiles = [fname[:-4] for fname in os.listdir(data_folder) if fname.endswith('.mp4')]
            print(subfiles)
            for sfile in subfiles:
                #if "c0028" in sfile:
                print sfile
                parsed_data = get_training_params(sfile, data_dir = full_data_path)
                write_syn_to_bin(parsed_data, os.path.join(output_folder, sfile) + ".bin")
          
