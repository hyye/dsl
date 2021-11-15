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
// Created by hyye on 12/28/19.
//

#ifndef DSL_YAML_LOADER_H_
#define DSL_YAML_LOADER_H_

#include <Eigen/Eigen>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

namespace dsl {
class YamlLoader {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  YamlLoader(std::string yaml_file);

  Eigen::Matrix3d eigen_R;
  Eigen::Vector3d eigen_T;
  std::string csv_file, pcd_file, mask_file;
  double fx = 0, fy = 0, cx = 0, cy = 0;
  double gamma1 = 0, gamma2 = 0, u0 = 0, v0 = 0, xi = 0;
  int image_width = 0, image_height = 0;
  Eigen::Matrix3f R_lc = Eigen::Matrix3f::Identity();
  Eigen::Vector3f t_lc = Eigen::Vector3f(0, 0, 0);
  Eigen::Affine3f T_lc = Eigen::Affine3f::Identity();
  Eigen::Affine3f initial_pose = Eigen::Affine3f::Identity();
  double intensity_scalar = 1.0;
  double rgb_scalar = 1.0;
  bool dso_baseline = false;
  int tracking_plot_level = 0;
  int fast_forward = 0;
  double retrack_threshold = 1.25;
  bool equal_hist = false;
  double surfel_size = 0.01;  // default surfel
  std::string output_folder = "";
  //  double init_translation_noise = 0.0;
  //  double init_rotation_noise = 0.0;
};
}  // namespace dsl

#endif  // DSL_YAML_LOADER_H_
