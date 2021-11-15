/**
* This file is part of Direct Sparse Localization (DSL).
*
* Copyright (C) 2021 Haoyang Ye <hy.ye at connect dot ust dot hk>,
* and Huaiyang Huang <hhuangat at connect dot use dot hk>,
* Robotics and Multiperception Lab (RAM-LAB <https://ram-lab.com>),
* The Hong Kong University of Science and Technology
*
* For more information please see <https://github.com/hyye/dsl>.
* If you use this code, please cite the respective publications as
* listed on the above websites.
*
* DSL is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSL is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSL.  If not, see <http://www.gnu.org/licenses/>.
*/
//
// Created by hyye on 2/27/20.
//

#ifndef DSL_HABITAT_SIM_READER_H_
#define DSL_HABITAT_SIM_READER_H_

#include <Eigen/Eigen>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <iomanip>
#include <sophus/se3.hpp>

namespace dsl {

class HabitatSimReader {
 public:
  typedef Sophus::SE3<double> SE3;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  enum CameraName { left = 0, right };

  HabitatSimReader(std::string _path, CameraName camera_name = left);

  std::string path, image_path, depth_path, pose_path;
  cv::Mat color_image;
  cv::Mat gray_image;
  cv::Mat depth_image;
  SE3 cam_pose;

  const std::vector<std::string> cameras = {"left", "right"};
  int w = 640, h = 480;
  std::vector<SE3> all_poses;
  Eigen::Matrix<float, 3, 3> K;

  bool ReadImageAndDepth(int index);
  bool ReadImage(int index);
};

}

#endif  // DSL_HABITAT_SIM_READER_H_
