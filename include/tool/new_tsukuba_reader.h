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
// Created by hyye on 11/8/19.
//

#ifndef DSL_NEW_TSUKUBA_READER_H_
#define DSL_NEW_TSUKUBA_READER_H_

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>
#include <iomanip>
#include <sophus/se3.hpp>


namespace dsl {

class NewTsukubaReader {
 public:
  typedef Sophus::SE3<double> SE3;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  enum SceneName { daylight = 0, flashlight, fluorescent, lamps };
  enum CameraName { left = 0, right };

  NewTsukubaReader(std::string _path, SceneName scene_name = fluorescent,
                   CameraName camera_name = left);

  std::string path, image_path, depth_path, pose_path;
  cv::Mat color_img;
  cv::Mat depth_img;
  SE3 cam_pose;
  const std::vector<std::string> scenes = {"daylight", "flashlight",
                                           "fluorescent", "lamps"};

  const std::vector<std::string> cameras = {"left", "right"};
  const std::vector<std::string> LR = {"L", "R"};
  const int w = 640, h = 480;
  std::vector<SE3> all_poses;
  Eigen::Matrix<float, 3, 3> K;

  bool ReadImageAndDepth(int index);
};

}  // namespace dsl

#endif  // DSL_NEW_TSUKUBA_READER_H_
