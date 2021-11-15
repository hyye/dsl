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
parser.add_argument('--file_name', type=str, help='input file name')
parser.add_argument('--odom_topic', type=str, help='odom topic name')
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

bagFile = args.file_name
odom_topic = args.odom_topic

print "reading file " + bagFile

bag = rosbag.Bag(bagFile)
bagContents = bag.read_messages()
bagName = bag.filename[:-4]

with open(bagName +'.csv', 'w') as outfile:
    csv_writer = csv.writer(outfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(['t'] + ['px'] + ['py'] + ['pz'] + ['ow'] + ['ox'] + ['oy'] + ['oz'])
    for topic, msg, t in bagContents:
        if topic == odom_topic:
            t = msg.header.stamp
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            csv_writer.writerow([t.to_nsec()] + [p.x] + [p.y] + [p.z] + [q.w] + [q.x] + [q.y] + [q.z])

in_path = os.path.dirname(bagFile)
data = pd.read_csv(bagName +'.csv', index_col=False)
data_path = os.path.join(in_path, 'images')
image_path = 'images'

NS2S = 1e-9

data[u't'] = data[u't'] * NS2S

lidar_time = np.array(data[u't'])
time_delay = 0.0 #-0.05

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

# aligned_indices = np.ones(num_images, dtype=int) * -1
aligned_indices = {}
image_time_filtered = {}

for time, filename in image_time_filename.items():
#     print(time, filename)
    if time < data_t_min or time > data_t_max:
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

# T_bc = np.matrix([[0, 0, 1, 0],
#                   [-1, 0, 0, 0],
#                   [0, -1, 0, 0],
#                   [0, 0, 0, 1]])

T_bc = np.matrix(np.eye(4))

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

    if pose_data[0] - time <= 0 and next_pose_data[0] - time >= 0:
        dt = next_pose_data[0] - pose_data[0]
        t = time - pose_data[0]
        ratio = t / dt
        
        dxyz = (np.array([next_pose_data[1], next_pose_data[2], next_pose_data[3]]) \
                - np.array([pose_data[1], pose_data[2], pose_data[3]])) * ratio +\
                np.array([pose_data[1], pose_data[2], pose_data[3]])
        q1 = [pose_data[5], pose_data[6], pose_data[7], pose_data[4]]
        q2 = [next_pose_data[5], next_pose_data[6], next_pose_data[7], next_pose_data[4]]
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

out_data.to_csv(os.path.join(in_path, 'cyt200226.csv'), index=False, float_format='%18.18f')