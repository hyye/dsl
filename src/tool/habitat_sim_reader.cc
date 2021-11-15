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

#include "tool/habitat_sim_reader.h"

namespace dsl {

HabitatSimReader::HabitatSimReader(
    std::string _path, dsl::HabitatSimReader::CameraName camera_name)
    : path(_path) {
  image_path = path + "/" + cameras[camera_name] + "/";
  depth_path = path + "/depth/";
  pose_path = path + "/trajectory.tum";
  std::ifstream pose_infile(pose_path);

  std::string line;
  while (std::getline(pose_infile, line)) {
    std::istringstream iss(line);
    Eigen::Matrix<double, 8, 1> vec_in;
    int i = 0;
    std::string s;

    while (iss >> s) {
      vec_in[i] = std::stod(s);
      std::cout << vec_in[i] << " ";
      ++i;
      if (iss.peek() == ' ') iss.ignore();
    }
    std::cout << std::endl;
    Eigen::Quaterniond q(vec_in.segment<4>(4));
    q.normalize();
    std::cout << q.coeffs() << std::endl;

    SE3 pose_in;
    pose_in.translation() = vec_in.segment<3>(1);
    pose_in.setRotationMatrix(q.toRotationMatrix());

    all_poses.emplace_back(pose_in);
  }

  K << 320, 0, 320, 0, 320, 240, 0, 0, 1;
}

bool HabitatSimReader::ReadImageAndDepth(int index) {
  std::stringstream ss;
  ss << std::setw(5) << std::setfill('0') << (index);
  std::string s = ss.str();
  std::string color_fn = image_path + s + ".png";
  std::string depth_fn = depth_path + s + ".xml";
  std::cout <<"color_fn: " << color_fn << std::endl;
  color_image = cv::imread(color_fn);
  cv::cvtColor(color_image, gray_image, cv::COLOR_BGR2GRAY);
  cv::FileStorage fs;
  fs.open(depth_fn, cv::FileStorage::READ);
  fs["depth"] >> depth_image;
  cam_pose = all_poses[index];
}

bool HabitatSimReader::ReadImage(int index) {
  std::stringstream ss;
  ss << std::setw(5) << std::setfill('0') << (index);
  std::string s = ss.str();
  std::string color_fn = image_path + s + ".png";
  std::cout <<"color_fn: " << color_fn << std::endl;
  color_image = cv::imread(color_fn);
  cv::cvtColor(color_image, gray_image, cv::COLOR_BGR2GRAY);
  cam_pose = all_poses[index];
}

}  // namespace dsl