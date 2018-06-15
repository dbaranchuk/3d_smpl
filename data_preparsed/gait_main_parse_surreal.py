# This file preparsed surreal data into binary files
# For fast tfrecord read-write
from load_surreal import get_training_params
import os
import numpy as np
import pickle as pkl
from write_utils import write_syn_to_bin, read_syn_to_bin

data_root_dir = '/home/local/data/cmc/synthetic/run0'
write_root_dir = '/home/local/data/cmc/synthetic/bin'
dataset_name = 'surreal'

subjects = ['02_01', '05_01', '06_01', 'ung_07_01', 'ung_07_03', 'ung_07_05', 'ung_07_08', 'ung_07_11', '08_01', '08_04', '08_05', '08_09', '08_11', '10_04', 'ung_12_01', 'ung_12_02', 'ung_12_03', '15_01', '26_01', '27_01', '37_01', '38_01', '38_02', '39_01', '39_02', '43_01', '45_01', 'ung_47_01', 'ung_49_01', '55_04', 'ung_74_01', 'ung_82_11', 'ung_82_12', 'ung_91_57', 'ung_104_02', 'ung_113_25', 'ung_132_18', 'ung_132_48', 'ung_120_20', 'ung_136_21', 'ung_139_28', '143_32']
directions = ['f', 'b']

def check_dir(path):
  if not os.path.exists(path):
    os.mkdir(path)

full_data_path = os.path.join(data_root_dir, dataset_name)
full_write_path = os.path.join(write_root_dir, dataset_name)

for subject in subjects:
    for direction in directions:
        filename = subject + "_" + direction

        if os.path.exists(os.path.join(full_data_path, filename)):
            data_folder = os.path.join(full_data_path, filename)
            output_folder = os.path.join(full_write_path, filename)
            check_dir(output_folder)

            subfiles = [fname[:-4] for fname in os.listdir(data_folder) if fname.endswith('.mp4')]
            for sfile in subfiles:
                print sfile
                parsed_data = get_training_params(sfile, data_dir = full_data_path)
                write_syn_to_bin(parsed_data, os.path.join(output_folder, sfile) + ".bin")
          
