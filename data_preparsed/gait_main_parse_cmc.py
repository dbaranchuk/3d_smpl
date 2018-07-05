# This file preparsed surreal data into binary files
# For fast tfrecord read-write
from load_cmc import get_training_params
import os
import numpy as np
import pickle as pkl
from write_utils import write_cmc_to_bin, read_cmc_to_bin

data_root_dir = '/home/local/data/cmc/real'
write_root_dir = '/home/local/data/cmc/real/bin'

is_test = True
if is_test:
    subjects = ['s1']
    modality = 'test/subdir'
else:
    pass

directions = ['f', 'b']


def check_dir(path):
  if not os.path.exists(path):
    os.mkdir(path)

full_write_path = os.path.join(write_root_dir, modality)

for subject in subjects:
    for direction in directions:
        filename = subject + "_" + direction
        if os.path.exists(os.path.join(data_root_dir , filename)):
            data_folder = os.path.join(data_root_dir , filename)
            check_dir(full_write_path)
            sfile = filename
            print sfile
            parsed_data = get_training_params(sfile, data_root_dir)
            write_cmc_to_bin(parsed_data, os.path.join(full_write_path, sfile) + "_" + direction + ".bin")
          
