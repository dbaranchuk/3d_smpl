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
data_path = "/home/local/data/cmc/synthetic/bin"
is_test = True

if is_test:
    modality = 'test'
else:
    modality = 'train'

data_path = os.path.join(data_path, modality)
print data_path

filename = "/home/local/mocap/tf_code/gait/" + dataset + "/" + modality + ".tfrecords"
convert_to_tfrecords_from_folder(data_path, filename, test=is_test, with_idx=with_idx, is_gait=True)
print filename
