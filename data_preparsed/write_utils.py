import numpy as np
import struct
import os
import json

w = 320
h = 240
def write_syn_to_bin(parsed_data, filename):
  # gender: 1 
  # beta: 100 x 10
  # pose: 100 x 72
  # f : 100 x2
  # R : 100 x 3
  # T : 100 x 3
  # J : 100x24x3
  # J_2d : 100 x 24 x2
  # image: 100 x 24 x 320 x 3 # np.unit8
  # seg: 100 x 240 x 320 #bool
  num_frames = parsed_data['pose'].shape[0]
  # gender[int32], num_frames[int32]
  with open(filename, "wb") as f_:
    f_.write(struct.pack('i', parsed_data['gender'])) 
    f_.write(struct.pack('i', num_frames)) 
    for frame_id in range(num_frames):
      beta = list(parsed_data['beta'][frame_id, :])
      pose = list(parsed_data['pose'][frame_id, :])
      f = list(parsed_data['f'][frame_id, :])
      R = list(parsed_data['R'][frame_id, :])
      T = list(parsed_data['T'][frame_id, :])
      J = list(np.reshape(parsed_data['J'][frame_id, :, :], [-1]))
      J_2d = list(np.reshape(parsed_data['J_2d'][frame_id, :, :], [-1]))
      J_2d_openpose = list(np.reshape(parsed_data['J_2d_openpose'][frame_id, :, :], [-1]))
      image = list(np.reshape(parsed_data['image'][frame_id, :, :, :].astype(np.float32), [-1]))
      params = beta + pose + f + R + T + J + J_2d + J_2d_openpose + image
      num_elements = len(params)
      f_.write(struct.pack('f' * num_elements, *params))
      seg = list(np.reshape(parsed_data['seg'][frame_id, :, :], [-1])) 
      f_.write(struct.pack('?' * h * w, *seg))
       

def read_syn_to_bin(filename, frame_id):
  with open(filename, 'rb') as f_:
    line = f_.read(4)
    gender = struct.unpack('i', line)[0] 
    line = f_.read(4)
    num_frames = struct.unpack('i', line)[0] 
    num_elements_in_line = 10 + 72 + 2 + 3 + 3 + 24 * 3 + 24 * 2 + 24 * 2 + h * w * 3
    # get to the head of requested frame
    _ = f_.read((4 * (num_elements_in_line) + h * w) * frame_id)
    line = f_.read(4 * num_elements_in_line)
    params = struct.unpack('f' * num_elements_in_line, line) 
    line = f_.read(1 * h * w)
    seg = struct.unpack('?' * h * w, line) 
    output = dict()
    output['gender'] = gender
    output['beta'] = params[:10]
    output['pose'] = params[10: 82]
    output['f'] = params[82:84]
    output['R'] = params[84:87]
    output['T'] = params[87:90]
    output['J'] = np.reshape(params[90:90 + 72], [24, 3])
    output['J_2d'] = np.reshape(params[162:162 + 48], [24, 2])
    output['J_2d_openpose'] = np.reshape(params[210:210 + 48], [24, 2])
    output['image'] = np.reshape(params[258:258 + h * w * 3], [h, w, 3])
    output['seg'] = np.reshape(seg, [h, w])
    return output

def read_openpose(filename, frame_id, annot_path):
    json_name = (os.path.join(annot_path, filename)+"_%012d_keypoints.json") % frame_id
    with open(json_name, 'r') as f:
        annot = json.load(f)
    if len(annot['people']) == 0:
        return np.zeros((24,2))
    joints = annot['people'][0]['pose_keypoints_2d']
    joints = np.array(joints).reshape((25, 3))[1:]
    assert(len(joints) == 24)
    visibility = joints[:,2]
    joints = joints[:,:2].astype('int32')
    # Permutate to SMPL
    #perm = np.array([7,11,8,16,12,9,17,13,10,18,22,20,0,19,14,15,4,1,5,2,6,3,21,23])
    perm = np.array([12,17,19,21,16,18,20,0,2,5,8,1,4,7,14,15,3,6,9,13,11,22,10,23])
    import cv2
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

    image = np.zeros((320,240,3))
    img_path = '/home/local/tmp'
    for i, joint in enumerate(joints[perm]):
        cv2.circle(image, tuple(joint), 2, colors[i], -1)
    cv2.imwrite(os.path.join(img_path, filename+'.jpg'), image)

    return joints[perm]
