import pandas as pd
import numpy as np
# import glob

import csv, rosbag
import os
from os import listdir
from os.path import isfile, join, abspath, splitext

import collections
from tf.transformations import quaternion_from_matrix, quaternion_matrix, quaternion_multiply, quaternion_slerp

from argparse import ArgumentParser

parser = ArgumentParser()
# parser.add_argument("-s",
#                     "--sqn_str",
#                     dest="sqn_str",
# #                     help="sqn_str")
# parser.add_argument("-i",
#                     "--in_path",
#                     dest="in_path",
#                     help="in_path")
parser.add_argument('--input', type=str, help='input file name')
parser.add_argument('--output', type=str, help='output file name')
args = parser.parse_args()

class VecTrans:
    def __init__(self, time, px, py, pz, ow, ox, oy, oz):
        self.time = time
        self.px = px
        self.py = py
        self.pz = pz
        self.ow = ow
        self.ox = ox
        self.oy = oy
        self.oz = oz
    def data(self):
        print(self.time, self.px, self.py, self.pz, self.ow, self.ox, self.oy, self.oz)

def vec2transform(item):
    transform = quaternion_matrix([item.ox, item.oy, item.oz, item.ow])
    transform[0:3, 3] = np.array([item.px, item.py, item.pz])
    return np.matrix(transform)

def transform2vec(transform):
    q = quaternion_from_matrix(transform)
    px = transform[0, 3]
    py = transform[1, 3]
    pz = transform[2, 3]
    ox = q[0]
    oy = q[1]
    oz = q[2]
    ow = q[3]
    tvec = VecTrans(0, px, py, pz, ow, ox, oy, oz)
    return tvec

infile = args.input
outfile = args.output

print("reading file " + infile)

in_path = os.path.dirname(os.path.dirname(infile))

image_path = 'color/images'
data_path = os.path.join(in_path, image_path)

data = pd.read_csv(infile, sep=' ', header=None)
data.columns = ['t', 'px', 'py', 'pz', 'ox', 'oy', 'oz', 'ow']

NS2S = 1e-9

# data[u't'] = data[u't'] * NS2S

lidar_time = np.array(data[u't'])
time_delay = 0.0

image_time_filename = dict([(float(splitext(f)[0]) * NS2S + time_delay, join(image_path, f)) for f in listdir(data_path) if isfile(join(data_path, f))])
image_time_filename = collections.OrderedDict(sorted(image_time_filename.items()))

data_t_max = data[u't'].max()
data_t_min = data[u't'].min()

image_time_filename

lidar_time[0]

data[u't'] - 1403715274.312143

last_time_index = image_count = 0
lidar_len = len(lidar_time)
num_images = len(image_time_filename)

print(lidar_len, num_images)

# aligned_indices = np.ones(num_images, dtype=int) * -1
aligned_indices = {}
image_time_filtered = {}

for time, filename in image_time_filename.items():
#     print(time, filename)
    if time < data_t_min or time > data_t_max:
        print(time, data_t_min, data_t_max)
        continue
    time_index = last_time_index
    time_next_index = time_index + 1
    valid = True
    for i in range(last_time_index, lidar_len):
        if i < (lidar_len - 1) and (i + 1) < (lidar_len - 1):
            if (lidar_time[i] - time) <= 0 and (lidar_time[i + 1] - time) > 0:
                time_index = i
                time_next_index = time_index + 1
                break
        else:
            print('must end', time_index)
            valid = False
            
    if valid:
        print(lidar_time[time_index] - time, lidar_time[i + 1] - time,\
            lidar_time[time_index], time, time_index)

        image_time_filtered[time] = filename
    
        aligned_indices[time] = (time_index, time_next_index) 
        image_count += 1
            
        last_time_index = time_index
#     print(time_index)

len(aligned_indices)

image_time_filtered = collections.OrderedDict(sorted(image_time_filtered.items()))


T_cb = np.matrix([[0.9999881386756897, 0.0036277398467063904, -0.00324101559817791, 0.014705127105116844],
                  [-0.003629520069807768, 0.9999932646751404, -0.0005435792845673859, 0.0001416415034327656],
                  [0.0032390218693763018, 0.0005553361843340099, 0.9999945759773254, 0.0002105139137711376],
                  [0, 0, 0, 1]])

T_bc = np.linalg.inv(T_cb)

# T_bc = np.matrix(np.eye(4))

times = list(image_time_filtered.keys())

i = 0
new_data = []
for time, filename in image_time_filtered.items():
    if (i >= len(aligned_indices)):
        break
    lidar_idx = aligned_indices[time][0]
    lidar_next_idx = aligned_indices[time][1]
    pose_data = data.iloc[lidar_idx]
    next_pose_data = data.iloc[lidar_next_idx]

    if pose_data[u't'] - time <= 0.0 and next_pose_data[u't'] - time >= 0.0 and next_pose_data[u't'] - time <= 0.02:
        dt = next_pose_data[u't'] - pose_data[u't']
        t = time - pose_data[u't']
        ratio = t / dt
        
        dxyz = (np.array([next_pose_data['px'], next_pose_data['py'], next_pose_data['pz']]) \
                - np.array([pose_data['px'], pose_data['py'], pose_data['pz']])) * ratio +\
                np.array([pose_data['px'], pose_data['py'], pose_data['pz']])
        q1 = [pose_data['ox'], pose_data['oy'], pose_data['oz'], pose_data['ow']]
        q2 = [next_pose_data['ox'], next_pose_data['oy'], next_pose_data['oz'], next_pose_data['ow']]
        dq = quaternion_slerp(q1, q2, ratio)
        
        px = dxyz[0]
        py = dxyz[1]
        pz = dxyz[2]
        ox = dq[0]
        oy = dq[1]
        oz = dq[2]
        ow = dq[3]
        
#         print(ratio, ow, ox, oy, oz)
        
        vec_v0vk = VecTrans(time, px, py, pz, ow, ox, oy, oz)
        T_v0vk = vec2transform(vec_v0vk)
        T_wck = T_v0vk * T_bc
        vec_wck = transform2vec(T_wck)
        
        new_row = [time, vec_wck.px, vec_wck.py, vec_wck.pz, vec_wck.ow,\
                   vec_wck.ox, vec_wck.oy, vec_wck.oz, filename]
        new_data.append(new_row)
        print(time)
    else:
        print(pose_data[0], next_pose_data[0], time)
    
    i += 1

out_data = pd.DataFrame(new_data, columns=['t', 'px', 'py', 'pz', 'ow', 'ox', 'oy', 'oz', 'filename'])

out_data.to_csv(os.path.join(in_path, outfile), index=False, float_format='%18.18f')