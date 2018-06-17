import numpy as np
import pickle
import os 
import random
from write_utils import read_syn_to_bin
import struct 
import sys
from tfrecord_utils import convert_to_tfrecords_from_folder
with_idx=True
dataset = "surreal"
data_path = "/home/local/data/surreal/SURREAL/data/bin"
is_test = False
data_path = os.path.join(data_path, dataset)
modalities = ['train', 'test', 'val']
runs = ['run0', 'run1', 'run2']

for modality in modalities:
    data_path = os.path.join(data_path, modality)
    for run in runs:
        data_path = os.path.join(data_path, run)
        if not os.path.exists(data_path):
            continue

        filename = "/home/local/mocap/tf_code/train/surreal_" + modality + "_" + run + ".tfrecords"
        convert_to_tfrecords_from_folder(data_path, filename, test=is_test, with_idx=with_idx, is_gait=False)
        print data_path, filename
