import scipy.io as sio
from matrix_utils import avg_joint_error, rotationMatrixToEulerAngles, eulerAnglesToRotationMatrix
import imageio
#import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
#from visualize import visualize_smpl_2d, visualize_smpl_3d, visualize_smpl_mesh, visualize_smpl_3d_mesh
from smpl_webuser.serialization import load_model


def draw_2d_joints(image, joints, name='vis.jpg'):
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


def get_training_params(filename, data_dir, direction=None):
  folder_name = filename
  if direction != None:
    filename += direction

  w = 320
  h = 240
  cap = imageio.get_reader(os.path.join(data_dir, folder_name, filename) + ".mp4")
  all_J_2d = np.load(os.path.join(data_dir, folder_name, filename) + "_reconstructed_2d.npy")
  all_J_2d[:, :, 0] = w - all_J_reconstruct_2d[:, :, 0]

  num_frames = all_J_2d.shape[0]
  all_image = np.zeros((num_frames, h, w, 3), dtype=np.uint8)

  for frame_id in range(num_frames):
    img = cap.get_data(frame_id)
    #draw_2d_joints(np.array(img), d2.T, name='/home/local/tmp/dir/vis'+str(frame_id)+'.jpg')
    all_image[frame_id, :, :, :] = np.fliplr(img)

  output = dict()
  output['J_2d'] = all_J_2d
  output['image'] = all_image
  return output
