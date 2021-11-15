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

#include "map_render/yaml_loader.h"
#include <boost/filesystem.hpp>
#include "util/util_common.h"

namespace dsl {

using boost::filesystem::path;

YamlLoader::YamlLoader(std::string yaml_file) {
  path path_yaml(yaml_file);

  cv::FileStorage fs_settings(yaml_file, cv::FileStorage::READ);

  fs_settings["image_width"] >> image_width;
  fs_settings["image_height"] >> image_height;

  cv::FileNode n = fs_settings["distortion_parameters"];
  n = fs_settings["projection_parameters"];
  fx = static_cast<double>(n["fx"]);
  fy = static_cast<double>(n["fy"]);
  cx = static_cast<double>(n["cx"]);
  cy = static_cast<double>(n["cy"]);

  if (!fs_settings["cata_projection_parameters"].empty()) {
    n = fs_settings["cata_projection_parameters"];
    gamma1 = static_cast<double>(n["gamma1"]);
    gamma2 = static_cast<double>(n["gamma2"]);
    u0 = static_cast<double>(n["u0"]);
    v0 = static_cast<double>(n["v0"]);
    xi = static_cast<double>(n["xi"]);
  }

  cv::Mat cv_R, cv_T;
  fs_settings["extrinsic_rotation"] >> cv_R;
  fs_settings["extrinsic_translation"] >> cv_T;
  cv::cv2eigen(cv_R, R_lc);
  cv::cv2eigen(cv_T, t_lc);

  T_lc.setIdentity();
  T_lc.linear() = R_lc;
  T_lc.translation() = t_lc;

  fs_settings["csv_file"] >> csv_file;
  fs_settings["pcd_file"] >> pcd_file;

  fs_settings["output_folder"] >> output_folder;

  path path_csv(csv_file);
  path path_pcd(pcd_file);

  if (path_csv.is_relative()) {
    csv_file = (path_yaml.parent_path() / path_csv).string();
  }

  if (path_pcd.is_relative()) {
    pcd_file = (path_yaml.parent_path() / path_pcd).string();
  }

  if (!fs_settings["mask_file"].empty()) {
    fs_settings["mask_file"] >> mask_file;
    path path_mask(mask_file);
    mask_file = (path_yaml.parent_path() / path_mask).string();
  } else {
    mask_file = "";
  }

  fs_settings["intensity_scalar"] >> intensity_scalar;
  if (!fs_settings["rgb_scalar"].empty()) {
    fs_settings["rgb_scalar"] >> rgb_scalar;
  }

  LOG(INFO) << "csv: " << csv_file;
  LOG(INFO) << "pcd: " << pcd_file;

  if (!fs_settings["dso_baseline"].empty()) {
    dso_baseline = (((int) fs_settings["dso_baseline"]) > 0);
  }

  if (!fs_settings["tracking_plot_level"].empty()) {
    fs_settings["tracking_plot_level"] >> tracking_plot_level;
  }

  if (!fs_settings["fast_forward"].empty()) {
    fs_settings["fast_forward"] >> fast_forward;
  }

  if (!fs_settings["retrack_threshold"].empty()) {
    fs_settings["retrack_threshold"] >> retrack_threshold;
  }

  if (!fs_settings["equal_hist"].empty()) {
    equal_hist = (((int) fs_settings["equal_hist"]) > 0);
  }

  if (!fs_settings["surfel_size"].empty()) {
    fs_settings["surfel_size"] >> surfel_size;
  }

  if (!fs_settings["initial_pose"].empty()) {
    cv::FileNode n = fs_settings["initial_pose"];
    double tx = static_cast<double>(n["tx"]);
    double ty = static_cast<double>(n["ty"]);
    double tz = static_cast<double>(n["tz"]);
    double qw = static_cast<double>(n["qw"]);
    double qx = static_cast<double>(n["qx"]);
    double qy = static_cast<double>(n["qy"]);
    double qz = static_cast<double>(n["qz"]);
    initial_pose.linear() = Eigen::Quaternionf(qw, qx, qy, qz).toRotationMatrix();
    initial_pose.translation() = Eigen::Vector3f(tx, ty, tz);
  }

  if (!fs_settings["initial_pose_tum"].empty()) {
    cv::Mat pose_mat;
    fs_settings["initial_pose_tum"] >> pose_mat;
    double tx = pose_mat.at<double>(0, 0);
    double ty = pose_mat.at<double>(0, 1);
    double tz = pose_mat.at<double>(0, 2);
    double qx = pose_mat.at<double>(0, 3);
    double qy = pose_mat.at<double>(0, 4);
    double qz = pose_mat.at<double>(0, 5);
    double qw = pose_mat.at<double>(0, 6);

    initial_pose.linear() = Eigen::Quaternionf(qw, qx, qy, qz).toRotationMatrix();
    initial_pose.translation() = Eigen::Vector3f(tx, ty, tz);
    LOG(INFO) << initial_pose.translation().transpose() << " "
              << Eigen::Quaternionf(initial_pose.rotation()).coeffs().transpose();
  }

  //  if (!fs_settings["init_translation_noise"].empty()) {
  //    fs_settings["init_translation_noise"] >> init_translation_noise;
  //  }
  //
  //  if (!fs_settings["init_rotation_noise"].empty()) {
  //    fs_settings["init_rotation_noise"] >> init_rotation_noise;
  //  }

  //  if (!fs_settings["tracking_debug_mode"].empty()) {
  //    tracking_debug_mode = (((int) fs_settings["tracking_debug_mode"]) > 0);
  //  }
}

}  // namespace dsl