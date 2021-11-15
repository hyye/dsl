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
// Created by hyye on 7/23/20.
//

#ifndef DSL_NET_VLAD_UTILS_H
#define DSL_NET_VLAD_UTILS_H

#include <opencv2/core.hpp>
#include <fstream>

namespace dsl::relocalization {

class NetVladUtils {
 public:
  static cv::Mat ReadVLADBinary(const std::string &file_name) {
    const int height = 1;
    const int width = 4096;
    // cout << file_name << endl;
    // allocate buffer
    const int buffer_size_ = sizeof(float) * height * width;
    char buffer_[buffer_size_];
    // open filestream && read buffer
    std::ifstream fs_bin_(file_name.c_str(), std::ios::binary);
    fs_bin_.read(buffer_, buffer_size_);
    fs_bin_.close();
    // construct depth map && return
    cv::Mat out = cv::Mat(cv::Size(width, height), CV_32FC1);
    std::memcpy(out.data, buffer_, buffer_size_);
    return out.clone();
  }
};

}

#endif // DSL_NET_VLAD_UTILS_H
