import numpy as np
import pickle
import os 
import random
from write_utils import read_cmc_to_bin
import struct 
import sys
from cmc_tfrecord_utils import convert_to_tfrecords_from_folder

with_idx=True
dataset = "real"
data_path = "/home/local/data/cmc/real/bin"
is_test = True
modality = 'test' if is_test else 'train'

data_path = os.path.join(data_path, modality)
print data_path

filename = "/home/local/mocap/tf_code/gait/" + dataset + "/" + modality + ".tfrecords"
convert_to_tfrecords_from_folder(data_path, filename, is_test=is_test, with_idx=with_idx, is_gait=True)
print filename
