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
data_path = "/home/local/mocap/output"
num_samples = 2
is_test = True
data_path = os.path.join(data_path, dataset)

#if train:
#  data_path = os.path.join(data_path, "train")
#else:
#  data_path = os.path.join(data_path, "test")

print data_path

#quo = sys.argv[1]
#print "quo", quo

subject = sys.argv[1]

#a = b
filename = "/home/local/mocap/tf_code/gait/surreal_" + subject + ".tfrecords"

convert_to_tfrecords_from_folder(data_path, filename, test=is_test, get_samples=num_samples, with_idx=with_idx)
print filename
