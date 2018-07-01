import numpy as np
import pickle
import os 
import random
from write_utils import read_syn_to_bin
import struct 
import sys
from surreal_tfrecord_utils import convert_to_tfrecords_from_folder
with_idx=True
dataset = "surreal"
data_path = "/home/local/data/cmc/synthetic/bin"
is_test = False
modality = 'test' if is_test else 'train_full_annot'

data_path = os.path.join(data_path, modality)
print data_path

filename = "/home/local/mocap/tf_code/gait/" + dataset + "/" + modality + ".tfrecords"
convert_to_tfrecords_from_folder(data_path, filename, is_test=is_test, with_idx=with_idx, is_gait=True)
print filename
