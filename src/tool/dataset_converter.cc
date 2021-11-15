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
// Created by hyye on 6/30/20.
//

#include "tool/dataset_converter.h"

namespace dsl {

std::vector<FrameShellWithFn> LoadTum(std::string pose_path) {
  std::vector<FrameShellWithFn> poses_with_fn;

  std::ifstream pose_infile(pose_path);

  std::string line;
  while (std::getline(pose_infile, line)) {
    std::istringstream iss(line);
    Eigen::Matrix<double, 8, 1> vec_in;
    int i = 0;
    std::string s;
    std::string filename;

    while (iss >> s) {
      if (i == 0) {
        filename = s.substr(0, 10) + s.substr(11, 9);
      }
      vec_in[i] = std::stod(s);
      // std::cout << vec_in[i] << " ";
      ++i;
      if (iss.peek() == ' ') iss.ignore();
    }
    // std::cout << std::endl;
    Eigen::Quaterniond q(vec_in.segment<4>(4));
    q.normalize();
    // std::cout << q.coeffs() << std::endl;

    SE3 pose_in;
    pose_in.translation() = vec_in.segment<3>(1);
    pose_in.setRotationMatrix(q.toRotationMatrix());

    FrameShellWithFn frame_shell;
    frame_shell.timestamp = vec_in(0);
    frame_shell.cam_to_world = pose_in;
    frame_shell.filename = filename;

    poses_with_fn.push_back(frame_shell);
  }
  return poses_with_fn;
}

std::string ToTum(std::string time, const SE3 &se3) {
  std::stringstream ss;
  Eigen::Quaterniond q = se3.unit_quaternion();
  Vec3 p = se3.translation();
  std::string time_converted;
  if (time.size() >= 18) {
    time_converted = time.substr(0, 10) + "." + time.substr(10, 9);
  } else {
    time_converted = time;
  }
  ss << time_converted << " " << p.x() << " "
     << p.y() << " " << p.z() << " " << q.x() << " " << q.y() << " " << q.z()
     << " " << q.w() << std::endl;
  return ss.str();
}

}