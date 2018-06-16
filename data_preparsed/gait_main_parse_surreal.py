# This file preparsed surreal data into binary files
# For fast tfrecord read-write
from load_surreal import get_training_params
import os
import numpy as np
import pickle as pkl
from write_utils import write_syn_to_bin, read_syn_to_bin

data_root_dir = '/home/local/data/cmc/synthetic/run0'
write_root_dir = '/home/local/data/cmc/synthetic/bin'
is_test = False

if is_test:
    subjects = ['ung_07_03', '08_05', '10_04', '39_02', 'ung_12_03']
    modality = 'test'
else:
    subjects = ['02_01', '05_01', '06_01', 'ung_07_01', 'ung_07_05', 'ung_07_08', 'ung_07_11', '08_01', '08_04', '08_09', '08_11', 'ung_12_01', 'ung_12_02', '15_01', '26_01', '27_01', '37_01', '38_01', '38_02', '39_01', '43_01', '45_01', 'ung_47_01', 'ung_49_01', '55_04', 'ung_74_01', 'ung_82_11', 'ung_82_12', 'ung_91_57', 'ung_104_02', 'ung_113_25', 'ung_132_18', 'ung_132_48', 'ung_120_20', 'ung_136_21', 'ung_139_28', '143_32']
    modality = 'train'

directions = ['f', 'b']


def check_dir(path):
  if not os.path.exists(path):
    os.mkdir(path)

full_data_path = data_root_dir #os.path.join(data_root_dir, modality)
full_write_path = os.path.join(write_root_dir, modality)

for subject in subjects:
    for direction in directions:
        filename = subject + "_" + direction
        if os.path.exists(os.path.join(full_data_path, filename)):
            data_folder = os.path.join(full_data_path, filename)
            output_folder = full_write_path#os.path.join(full_write_path, filename)
            check_dir(output_folder)
            subfiles = [fname[:-4] for fname in os.listdir(data_folder) if fname.endswith('.mp4')]
            for sfile in subfiles:
                print sfile
                #### FIX ####
                parsed_data = get_training_params(sfile, full_data_path, direction)
                write_syn_to_bin(parsed_data, os.path.join(output_folder, sfile) + ".bin")
          
