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
// Created by hyye on 5/13/20.
//

#ifndef DSL_DEPTH_GENERATOR_FROM_MAP_H_
#define DSL_DEPTH_GENERATOR_FROM_MAP_H_

#include <fstream>
#include <string>
#include <vector>
#include <sophus/se3.hpp>

void LoadTumToPoses(std::string pose_path, std::vector<Sophus::SE3<double>> &all_poses) {
  all_poses.clear();

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

    Sophus::SE3<double> pose_in;
    pose_in.translation() = vec_in.segment<3>(1);
    pose_in.setRotationMatrix(q.toRotationMatrix());

    all_poses.emplace_back(pose_in);
  }
}

#endif //DSL_DEPTH_GENERATOR_FROM_MAP_H_
