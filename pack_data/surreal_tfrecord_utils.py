import numpy as np
import pickle
import os 
import random
from write_utils import read_syn_to_bin
import struct 
from tqdm import tqdm
import tensorflow as tf
import math 
import scipy.misc
import sys
import cv2

sys.path.insert(0, '../')
from chamfer_utils import get_chamfer

def get_file_list(data_path, quo=0, test=False):
    files = []
    num = 0
    for folder in tqdm(os.listdir(data_path)):
        num += 1
        p = os.path.join(data_path, folder)
        for filename in os.listdir(p):
            with open(os.path.join(p, filename), 'rb') as f_:
                line = f_.read(4) # gender
                line = f_.read(4) # num_frames
                num_frames = struct.unpack('i', line)[0]
                print filename, "nframes", num_frames

                for frame_id in range(0, num_frames-1, 2):
                    if test or (not test and frame_id % 2 == 0): 
                        files.append(os.path.join(p, filename) + "#" + str(frame_id))
    print("number of folder", num)
    print("number of files", len(files))
    return files


def convert_to_npz_from_folder(folder_name, npz_filename, get_samples=None, quo=0):
  files = get_file_list(folder_name, quo)
  random.shuffle(files)

  num_files = len(files)
  num_frames = 2
  image_size_h = 240
  image_size_w = 320

  keypoints_num = 24
  bases_num = 10
  print("num_files", num_files)
  if not get_samples:
    get_samples = num_files
  data_pose = np.zeros((get_samples, num_frames, keypoints_num, 3))
  data_T = np.zeros((get_samples, num_frames, 3))
  data_R = np.zeros((get_samples, num_frames, 3))
  data_beta = np.zeros((get_samples, num_frames, bases_num))
  data_J = np.zeros((get_samples, num_frames, keypoints_num, 3))
  data_J_2d = np.zeros((get_samples, num_frames, keypoints_num, 2))
  data_image = np.zeros((get_samples, num_frames, image_size_h, image_size_w, 3), dtype = np.uint8)
  data_seg = np.zeros((get_samples, num_frames, image_size_h, image_size_w), dtype=np.bool_)
  data_f = np.zeros((get_samples, num_frames, 2))
  data_gender = np.zeros((get_samples), dtype=np.int32)

  for sample_id in tqdm(range(get_samples)):
    filename, t = files[sample_id].split("#")
    output = dict()
    output[0] = read_syn_to_bin(filename, int(t))
    output[1] = read_syn_to_bin(filename, int(t) + 1)

    for frame_id in range(num_frames): 
      pose = np.reshape(output[frame_id]['pose'], [keypoints_num, 3]) 
      data_pose[sample_id, frame_id, :, :] = pose
      data_gender[sample_id] = int(output[frame_id]['gender'])
      data_T[sample_id, frame_id, :] = output[frame_id]['T']
      data_R[sample_id, frame_id, :] = output[frame_id]['R']
      data_beta[sample_id, frame_id, :] = output[frame_id]['beta']
      data_J[sample_id, frame_id, :, :] = output[frame_id]['J']
      data_J_2d[sample_id, frame_id, :, :] = output[frame_id]['J_2d']
      data_image[sample_id, frame_id, :, :, :] = output[frame_id]['image']
      #print np.max(output['seg'][sample_id, :, :])
      data_seg[sample_id, frame_id, :, :] = output[frame_id]['seg']
      data_f[sample_id, frame_id, :] = output[frame_id]['f']


  np.savez(npz_filename, pose=data_pose, T=data_T, R=data_R, beta=data_beta, J=data_J,
           J_2d=data_J_2d, image=data_image, seg=data_seg, f=data_f, gender=data_gender)


def loadBatchSurreal_fromString(file_string, image_size=128, num_frames=2, keypoints_num=24, bases_num=10, chamfer_scale=0.5, is_gait=False):
  filename, t = file_string.split("#")
  output = dict()
  output[0] = read_syn_to_bin(filename, int(t))
  output[1] = read_syn_to_bin(filename, int(t) + 1)

  data_pose = np.zeros((num_frames, keypoints_num * 3))
  data_T = np.zeros((num_frames, 3))
  data_R = np.zeros((num_frames, 6))
  data_beta = np.zeros((num_frames, bases_num))
  data_J = np.zeros((num_frames, keypoints_num, 3))
  data_J_2d = np.zeros((num_frames, keypoints_num, 2))
  data_J_2d_openpose = np.zeros((num_frames, keypoints_num, 2))
  data_image = np.zeros((num_frames, image_size, image_size, 3))
  data_seg = np.zeros((num_frames, image_size, image_size))
  small_image_size = int(chamfer_scale * image_size)
  data_chamfer = np.zeros((num_frames, small_image_size, small_image_size))
  data_f = np.zeros((num_frames, 2))
  data_c = np.zeros((num_frames, 2))
  data_resize_scale = np.zeros((num_frames))
  data_gender = np.zeros(())
  # Cropping
  old_2d_center = np.array([(320 - 1)/2.0, (240-1)/2.0])

  # Use keypoint 0 in frame1 as center
  J_2d = output[0]['J_2d']
  ignore_joints = [3,6,9,13,14,15,22,23]

  new_2d_center = np.round(J_2d[0, :] + 10 * (np.random.uniform((2)) - 1)) + 0.5*np.ones((2))
  s = 1.25 #+ 0.1 * np.random.rand()

  crop_size = np.round(s * np.max(np.abs(J_2d - np.reshape(new_2d_center, [1, 1, -1]))))
  new_image_size = int(2*crop_size)
  x_min = int(math.ceil(new_2d_center[0] - crop_size)) 
  x_max = int(math.floor(new_2d_center[0] + crop_size)) 
  y_min = int(math.ceil(new_2d_center[1] - crop_size))
  y_max = int(math.floor(new_2d_center[1] + crop_size))
  resize_scale = float(image_size)/(crop_size * 2.0) 
  data_resize_scale[:] = resize_scale
  new_origin = np.array([x_min, y_min]) 
  data_c[:, :] = np.reshape(old_2d_center - new_origin, [-1, 2])
  
  for frame_id in range(num_frames): 
      data_pose[frame_id, :] = output[frame_id]['pose']      
      data_gender = int(output[frame_id]['gender'])
      data_R[frame_id, :3] = np.sin(output[frame_id]['R'])
      data_R[frame_id, 3:6] = np.cos(output[frame_id]['R'])
      data_beta[frame_id, :] = output[frame_id]['beta']
      data_f[frame_id, :] = output[frame_id]['f']
      data_T[frame_id, :] = output[frame_id]['T']
      data_J[frame_id, :, :] = output[frame_id]['J']
      data_J_2d[frame_id, :, :] = resize_scale * (output[frame_id]['J_2d'] - np.reshape(new_origin, [1, -1]))

      # Preprocess J_2d_openpose
      J_2d_openpose = resize_scale * (output[frame_id]['J_2d_openpose'] - np.reshape(new_origin, [1, -1]))
      indices = np.where(J_2d_openpose.sum(1) < 0)[0]
      J_2d_openpose[indices] = np.zeros((len(indices), 2))
      for i, joint in enumerate(J_2d_openpose):
          if joint[0] == 0 and joint[1] == 0:
              print("hui")
      data_J_2d_openpose[frame_id, : , :] = J_2d_openpose

      # crop image
      image = output[frame_id]['image']
      h, w, _ = image.shape
      img_x_min = max(x_min, 0) 
      img_x_max = min(x_max, w -1)
      img_y_min = max(y_min, 0)
      img_y_max = min(y_max, h -1)

      crop_image = np.zeros((new_image_size, new_image_size, 3), dtype=np.float32) 
      crop_image[max(0, -y_min):max(0, -y_min) + img_y_max - img_y_min + 1, \
                 max(0, -x_min):max(0, -x_min) + img_x_max - img_x_min + 1, :] \
                 = image[img_y_min:img_y_max + 1, img_x_min:img_x_max +1, :] 
      data_image[frame_id, :, :, :] = scipy.misc.imresize(crop_image, [image_size, image_size])
      #draw_2d_joints_surreal(data_image[frame_id, :, :, :], data_J_2d_openpose[frame_id, :, :].astype('int32'), name='/home/local/tmp/src'+str(t)+str(frame_id)+'.jpg')
      
      seg_float = output[frame_id]['seg'].astype(np.float32)
      crop_seg = np.zeros((new_image_size, new_image_size, 3), dtype=np.float32)
      crop_seg[max(0, -y_min):max(0, -y_min) + img_y_max - img_y_min + 1, \
               max(0, -x_min):max(0, -x_min) + img_x_max - img_x_min + 1] \
               = np.expand_dims(seg_float[img_y_min:img_y_max + 1, img_x_min:img_x_max +1], 2)
      seg = scipy.misc.imresize(crop_seg, [image_size, image_size])
      seg[seg < 0.5] = 0
      seg[seg >= 0.5] = 1

      data_seg[frame_id, :, :] = seg[:, :, 0]
      data_chamfer[frame_id, :, :], _, _ = get_chamfer(seg[:,:,0], chamfer_scale)
  return data_pose, data_T, data_R, data_beta, data_J, data_J_2d, data_J_2d_openpose, data_image/255.0, data_seg, data_f, data_chamfer, data_c, data_gender, data_resize_scale


def _floatList_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))
def _intList_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def convert_to_tfrecords_from_folder(folder_name, tf_filename, get_samples=None, is_test=False, quo=0, with_idx=False, shuffle=True, is_gait=False):
    files = get_file_list(folder_name, quo, test=is_test)
    shuffle = not is_test
    if shuffle:
        random.shuffle(files)
    num_files = len(files)
    crop_image_size = 128
    keypoints_num = 24
    bases_num = 10
    if not get_samples:
        get_samples = num_files
    print("total samples", get_samples)
  
    writer = tf.python_io.TFRecordWriter(tf_filename)
    filename_to_idx = dict()
    counters = dict()
    for sample_id in tqdm(range(get_samples)):
        pose, T, R, beta, J, J_2d, J_2d_openpose, image, seg, f, chamfer, c, gender, resize_scale = loadBatchSurreal_fromString(files[sample_id], crop_image_size, is_gait=is_gait)
        scipy.misc.imsave('../tmp/image_' + str(sample_id) + '.png', image[0, :, :, :])
        feature={
            'pose': _floatList_feature(pose.flatten()),
            'beta': _floatList_feature(beta.flatten()),
            'T': _floatList_feature(T.flatten()),
            'R': _floatList_feature(R.flatten()),
            'J': _floatList_feature(J.flatten()),
            'J_2d': _floatList_feature(J_2d.flatten()),
            'J_2d_openpose': _floatList_feature(J_2d_openpose.flatten()),
            'image': _floatList_feature(image.flatten()),
            'seg': _floatList_feature(seg.flatten()),
            'f': _floatList_feature(f.flatten()),
            'chamfer': _floatList_feature(chamfer.flatten()),
            'c': _floatList_feature(c.flatten()),
            'resize_scale': _floatList_feature(resize_scale.flatten()),
            'gender': _intList_feature([gender])
        }
        if with_idx:
            filename, t = files[sample_id].split("#")
            if not filename in filename_to_idx.keys():
                counters[filename] = 0
                max_idx = -1
                for fname in filename_to_idx.keys():
                    idx = filename_to_idx[fname]
                    if idx > max_idx:
                        max_idx = idx
                filename_to_idx[filename] = max_idx + 1
            idx = filename_to_idx[filename]
            counters[filename] += 1
            feature['idx'] = _intList_feature([idx])#[sample_id])

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

    for i in counters.keys():
        print(counters[i])
    writer.close()


def read_and_decode_surreal(tfrecord_file):
  num_frames = 2
  image_size = 128
  keypoints_num = 24
  bases_num = 10
  chamfer_scale = 0.5
  small_image_size = int(chamfer_scale * image_size)
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(tfrecord_file)
  feature = tf.parse_single_example(
    serialized_example,  
    features={
      'pose': tf.FixedLenFeature([num_frames*keypoints_num*3], tf.float32),
      'beta': tf.FixedLenFeature([num_frames*bases_num], tf.float32),
      'T': tf.FixedLenFeature([num_frames*3], tf.float32),
      'R': tf.FixedLenFeature([num_frames*6], tf.float32),
      'J': tf.FixedLenFeature([num_frames*keypoints_num*3], tf.float32),
      'J_2d': tf.FixedLenFeature([num_frames*keypoints_num*2], tf.float32),
      'J_2d_openpose': tf.FixedLenFeature([num_frames*keypoints_num*2], tf.float32),
      'image': tf.FixedLenFeature([num_frames*image_size*image_size*3], tf.float32),
      'seg': tf.FixedLenFeature([num_frames*image_size*image_size], tf.float32),
      'f': tf.FixedLenFeature([num_frames*2], tf.float32),
      'chamfer': tf.FixedLenFeature([num_frames*small_image_size*small_image_size], tf.float32),
      'c': tf.FixedLenFeature([num_frames*2], tf.float32),
      'resize_scale': tf.FixedLenFeature([num_frames], tf.float32),
      'gender': tf.FixedLenFeature([], tf.int64),
    })
  feature['pose'] = tf.reshape(feature['pose'], [num_frames, keypoints_num, 3])
  feature['beta'] = tf.reshape(feature['beta'], [num_frames, bases_num])
  feature['T'] = tf.reshape(feature['T'], [num_frames, 3])
  feature['R'] = tf.reshape(feature['R'], [num_frames, 6])
  feature['J'] = tf.reshape(feature['J'], [num_frames, keypoints_num, 3])
  feature['J_2d'] = tf.reshape(feature['J_2d'], [num_frames, keypoints_num, 2])
  feature['J_2d_openpose'] = tf.reshape(feature['J_2d_openpose'], [num_frames, keypoints_num, 2])
  feature['image'] = tf.reshape(feature['image'], [num_frames, image_size, image_size, 3])
  feature['seg'] = tf.reshape(feature['seg'], [num_frames, image_size, image_size])
  feature['chamfer'] = tf.reshape(feature['chamfer'], [num_frames, small_image_size, small_image_size])
  feature['c'] = tf.reshape(feature['c'], [num_frames, 2])
  feature['f'] = tf.reshape(feature['f'], [num_frames, 2])
  feature['resize_scale'] = tf.reshape(feature['resize_scale'], [num_frames])
  return feature['pose'], feature['beta'], feature['T'], feature['R'], feature['J'], feature['J_2d'], feature['J_2d_openpose'], feature['image'], feature['seg'], feature['chamfer'], feature['c'], feature['f'], feature['resize_scale'], feature['gender']
 

def read_and_decode_surreal_with_idx(tfrecord_file):
  num_frames = 2
  image_size = 128
  keypoints_num = 24
  bases_num = 10
  chamfer_scale = 0.5
  small_image_size = int(chamfer_scale * image_size)
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(tfrecord_file)
  feature = tf.parse_single_example(
    serialized_example,  
    features={
      'pose': tf.FixedLenFeature([num_frames*keypoints_num*3], tf.float32),
      'beta': tf.FixedLenFeature([num_frames*bases_num], tf.float32),
      'T': tf.FixedLenFeature([num_frames*3], tf.float32),
      'R': tf.FixedLenFeature([num_frames*6], tf.float32),
      'J': tf.FixedLenFeature([num_frames*keypoints_num*3], tf.float32),
      'J_2d': tf.FixedLenFeature([num_frames*keypoints_num*2], tf.float32),
      'J_2d_openpose': tf.FixedLenFeature([num_frames*keypoints_num*2], tf.float32),
      'image': tf.FixedLenFeature([num_frames*image_size*image_size*3], tf.float32),
      'seg': tf.FixedLenFeature([num_frames*image_size*image_size], tf.float32),
      'f': tf.FixedLenFeature([num_frames*2], tf.float32),
      'chamfer': tf.FixedLenFeature([num_frames*small_image_size*small_image_size], tf.float32),
      'c': tf.FixedLenFeature([num_frames*2], tf.float32),
      'resize_scale': tf.FixedLenFeature([num_frames], tf.float32),
      'gender': tf.FixedLenFeature([], tf.int64),
      'idx': tf.FixedLenFeature([], tf.int64),
    })
  feature['pose'] = tf.reshape(feature['pose'], [num_frames, keypoints_num, 3])
  feature['beta'] = tf.reshape(feature['beta'], [num_frames, bases_num])
  feature['T'] = tf.reshape(feature['T'], [num_frames, 3])
  feature['R'] = tf.reshape(feature['R'], [num_frames, 6])
  feature['J'] = tf.reshape(feature['J'], [num_frames, keypoints_num, 3])
  feature['J_2d'] = tf.reshape(feature['J_2d'], [num_frames, keypoints_num, 2])
  feature['J_2d_openpose'] = tf.reshape(feature['J_2d_openpose'], [num_frames, keypoints_num, 2])
  feature['image'] = tf.reshape(feature['image'], [num_frames, image_size, image_size, 3])
  feature['seg'] = tf.reshape(feature['seg'], [num_frames, image_size, image_size])
  feature['chamfer'] = tf.reshape(feature['chamfer'], [num_frames, small_image_size, small_image_size])
  feature['c'] = tf.reshape(feature['c'], [num_frames, 2])
  feature['f'] = tf.reshape(feature['f'], [num_frames, 2])
  feature['resize_scale'] = tf.reshape(feature['resize_scale'], [num_frames])
  return feature['pose'], feature['beta'], feature['T'], feature['R'], feature['J'], feature['J_2d'], feature['J_2d_openpose'], feature['image'], feature['seg'], feature['chamfer'], feature['c'], feature['f'], feature['resize_scale'], feature['gender'], feature['idx']


def inputs_surreal(tf_filenames, batch_size, shuffle=True, openpose=False):
  with tf.name_scope('surreal_input'):
    filename_queue = tf.train.string_input_producer(tf_filenames)
    pose, beta, T, R, J, J_2d, J_2d_openpose, image, seg, chamfer, c, f, resize_scale, gender = read_and_decode_surreal(filename_queue)

    if openpose:
        J_2d = J_2d_openpose

    if not shuffle:
        return tf.train.batch([pose, beta, T, R, J, J_2d, image, seg, chamfer, c, f, resize_scale, gender], batch_size=batch_size, num_threads=2)
    else:
        return tf.train.shuffle_batch([pose, beta, T, R, J, J_2d, image, seg, chamfer, c, f, resize_scale, gender], batch_size=batch_size, num_threads=2, capacity=5000,min_after_dequeue=2000)


def inputs_surreal_with_idx(tf_filenames, batch_size, shuffle=True, openpose=False):
  with tf.name_scope('surreal_input'):
    filename_queue = tf.train.string_input_producer(tf_filenames, shuffle=shuffle)
    pose, beta, T, R, J, J_2d, J_2d_openpose, image, seg, chamfer, c, f, resize_scale, gender, idx = read_and_decode_surreal_with_idx(filename_queue)

    if openpose:
        J_2d = J_2d_openpose

    if not shuffle:
      return tf.train.batch([pose, beta, T, R, J, J_2d, image, seg, chamfer, c, f, resize_scale, gender, idx], batch_size=batch_size, num_threads=2) #,capacity=80,min_after_dequeue=50)
    else: 
      return tf.train.shuffle_batch([pose, beta, T, R, J, J_2d, image, seg, chamfer, c, f,resize_scale, gender, idx], batch_size=batch_size, num_threads=2, capacity=80, min_after_dequeue=50)


def draw_2d_joints_surreal(image, joints, name):
    left_leg = [1, 4, 7, 10]
    left_hand = [13, 16, 18, 20, 22]
    right_leg = [2, 5, 8, 11]
    right_hand = [14, 17, 19, 21, 23]
    spine = [0, 3, 6, 9, 12, 15]

    colors = {}
    for i in left_leg:
        colors[i] = (0, 255, 255)
    for i in right_leg:
        colors[i] = (0, 255, 0)
    for i in left_hand:
        colors[i] = (255, 0, 0)
    for i in right_hand:
        colors[i] = (0, 0, 255)
    for i in spine:
        colors[i] = (128, 128, 0)

    img_path = '/home/local/tmp'
    for i, joint in enumerate(joints):
        cv2.circle(image, tuple(joint), 2, colors[i], -1)
    cv2.imwrite(os.path.join(img_path, name), image)

