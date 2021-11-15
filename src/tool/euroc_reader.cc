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

#include "tool/euroc_reader.h"
#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

namespace dsl {

EurocReader::EurocReader(std::string _path, std::string config_filename) : path(_path) {
  image_path = path + "/" + "images";
  depth_path = path + "/" + "depths";

  for (const auto& entry : fs::directory_iterator(image_path)) {
    filenames.emplace_back(entry.path().stem().string());
    if (file_ext == "") {
      file_ext = entry.path().extension().string();
    }
  }

  std::sort(filenames.begin(), filenames.end());

  config_path = path + "/" + config_filename;
  cv::FileStorage fs(config_path, cv::FileStorage::READ);
  float fx = fs["projection_parameters"]["fx"];
  float fy = fs["projection_parameters"]["fy"];
  float cx = fs["projection_parameters"]["cx"];
  float cy = fs["projection_parameters"]["cy"];
  K << fx, 0, cx, 0, fy, cy, 0, 0, 1;

  w = fs["image_width"];
  h = fs["image_height"];
}

void EurocReader::ReadImage(int idx) {
  std::string fn = image_path + "/" + filenames[idx] + file_ext;
  gray_image = cv::imread(fn, CV_LOAD_IMAGE_GRAYSCALE);
}

int EurocReader::ReadImage(const std::string& filename) {
  auto it = std::find(filenames.begin(), filenames.end(), filename);
  if (it != filenames.end()) {
    int idx_in_filenames =
        std::distance(filenames.begin(), it);

    std::string fn = image_path + "/" + filename + file_ext;
    gray_image = cv::imread(fn, CV_LOAD_IMAGE_GRAYSCALE);
    return idx_in_filenames;
  } else {
    return -1;
  }
}

void EurocReader::ReadDepth(int idx) {
  std::string fn = depth_path + "/" + filenames[idx] + ".xml";
  std::cout << fn;
  dist_image = cv::imread(fn);
  cv::FileStorage fs(fn, cv::FileStorage::READ);
  fs["depth"] >> dist_image;
}

void EurocReader::ReadImageAndDist(int idx) {
  ReadImage(idx);
  ReadDepth(idx);
}

}  // namespace dsl