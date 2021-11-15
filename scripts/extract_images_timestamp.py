#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2016 Massachusetts Institute of Technology

"""Extract images from a rosbag.
"""

import os
import argparse

import cv2

import rosbag
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def main():
    """Extract a folder of images from a rosbag.
    """
    parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    parser.add_argument("--bag_file", help="Input ROS bag.")
    parser.add_argument("--output_dir", help="Output directory.")
    parser.add_argument("--image_topic", help="Image topic.")

    args = parser.parse_args()

    print "Extract images from %s on topic %s into %s" % (args.bag_file,
                                                          args.image_topic, args.output_dir)

    bag = rosbag.Bag(args.bag_file, "r")
    bridge = CvBridge()
    count = 0
    for topic, msg, t in bag.read_messages(topics=[args.image_topic]):

        # import IPython
        # IPython.embed()
        # break

        if "Compressed" in str(type(msg)):
            np_arr = np.fromstring(msg.data, np.uint8)
            cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        else:
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

        # import IPython
        # IPython.embed()
        # break
        # cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

        # output_img_path = str(os.path.join(args.output_dir, str(msg.header.stamp.secs) + "." + str(msg.header.stamp.nsecs).zfill(9) +".jpg"))
        output_img_path = str(os.path.join(args.output_dir, str(msg.header.stamp.to_nsec()) +".png"))
        cv2.imwrite(output_img_path, cv_img)
        print "Wrote image %i %s" % (count, output_img_path) 

        count += 1

    bag.close()

    return

if __name__ == '__main__':
    main()
