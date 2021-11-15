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
// Created by hyye on 7/3/20.
//

#ifndef DSL_CONVERTER_H
#define DSL_CONVERTER_H

#include <opencv2/core.hpp>
#include "util/num_type.h"

namespace dsl {

namespace relocalization {

class Converter {
 public:
  static std::vector<cv::Mat> toDescriptorVector(const cv::Mat &Descriptors);

  static SE3 toSE3Quat(const cv::Mat &cvT);

  static cv::Mat toCvMat(const SE3 &SE3);
  static cv::Mat toCvMat(const Sim3 &sim3);
  static cv::Mat toCvMat(const Eigen::Matrix<double,4,4> &m);
  static cv::Mat toCvMat(const Eigen::Matrix3d &m);
  static cv::Mat toCvMat(const Eigen::Matrix<double,3,1> &m);
  static cv::Mat toCvSE3(const Eigen::Matrix<double,3,3> &R, const Eigen::Matrix<double,3,1> &t);

  static Eigen::Matrix<double,3,1> toVector3d(const cv::Mat &cvVector);
  static Eigen::Matrix<double,3,1> toVector3d(const cv::Point3f &cvPoint);
  static Eigen::Matrix<double,3,3> toMatrix3d(const cv::Mat &cvMat3);

  static std::vector<float> toQuaternion(const cv::Mat &M);
};

}

}

#endif // DSL_CONVERTER_H
