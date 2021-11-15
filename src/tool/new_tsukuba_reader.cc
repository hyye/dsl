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

#include "tool/new_tsukuba_reader.h"

namespace dsl {

NewTsukubaReader::NewTsukubaReader(
    std::string _path, dsl::NewTsukubaReader::SceneName scene_name,
    dsl::NewTsukubaReader::CameraName camera_name)
    : path(_path) {
  image_path = path + "/illumination/" + scenes[scene_name] + "/" +
               cameras[camera_name] + "/tsukuba_" + scenes[scene_name] + "_" +
               LR[camera_name] + "_";
  depth_path = path + "/groundtruth/depth_maps/" + cameras[camera_name] +
               "/tsukuba_depth_" + LR[camera_name] + "_";
  pose_path = path + "/groundtruth/camera_track.txt";
  std::ifstream pose_infile(pose_path);

  std::string line;
  while (std::getline(pose_infile, line)) {
    std::istringstream iss(line);
    Eigen::Matrix<double, 6, 1> vec_in;
    int i = 0;
    std::string s;

    while (iss >> s) {
      vec_in[i] = std::stod(s);
      // std::cout << vec_in[i] << " ";
      ++i;
      if (iss.peek() == ',') iss.ignore();
    }
    // std::cout << std::endl;
    vec_in.head<3>() /= 100.0;
    vec_in[1] = -vec_in[1];
    vec_in[2] = -vec_in[2];
    vec_in.tail<3>() *= (M_PI / 180.0);

    Eigen::Matrix3d R(Eigen::AngleAxisd(-vec_in[5], Eigen::Vector3d::UnitZ()) *
                      Eigen::AngleAxisd(-vec_in[4], Eigen::Vector3d::UnitY()) *
                      Eigen::AngleAxisd(vec_in[3], Eigen::Vector3d::UnitX()));
    SE3 pose_in;
    pose_in.translation() = vec_in.head<3>();
    pose_in.setRotationMatrix(R);

    all_poses.emplace_back(pose_in);
  }

  K << 615, 0, 320, 0, 615, 240, 0, 0, 1;
}

bool NewTsukubaReader::ReadImageAndDepth(int index) {
  std::stringstream ss;
  ss << std::setw(5) << std::setfill('0') << (index + 1);
  std::string s = ss.str();
  std::string color_fn = image_path + s + ".png";
  std::string depth_fn = depth_path + s + ".xml";
  color_img = cv::imread(color_fn);
  cv::FileStorage fs;
  fs.open(depth_fn, cv::FileStorage::READ);
  fs["depth"] >> depth_img;
  depth_img /= 100.0;
  cam_pose = all_poses[index];
}

}
