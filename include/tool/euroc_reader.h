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
// Created by hyye on 12/19/19.
//

#ifndef DSL_EUROC_READER_H_
#define DSL_EUROC_READER_H_

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <sophus/se3.hpp>

namespace dsl {

class EurocReader {
 public:
  EurocReader(std::string _path, std::string config_filename="config.yaml");

  void ReadImageAndDist(int idx);
  void ReadImage(int idx);
  int ReadImage(const std::string& filename);
  void ReadDepth(int idx);

  std::string path;
  std::string image_path;
  std::string depth_path;
  std::string config_path;
  std::vector<std::string> filenames;
  std::string file_ext = "";

  cv::Mat gray_image;
  cv::Mat dist_image;

  Eigen::Matrix3f K;
  int w, h;

};

}

#endif //DSL_EUROC_READER_H_
