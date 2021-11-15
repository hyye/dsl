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
// Created by hyye on 7/14/20.
//

#ifndef DSL_REPROJECTION_FUNCTOR_H
#define DSL_REPROJECTION_FUNCTOR_H

#include <ceres/ceres.h>
#include "dsl_common.h"

namespace dsl::relocalization {

class ReprojectionFunctor : public ceres::SizedCostFunction<2, 6> {
 public:
  ReprojectionFunctor() {}
  ReprojectionFunctor(ReprojectionFunctor &functor);
  virtual bool Evaluate(double const *const *parameters, double *residuals,
                        double **jacobians) const;
  void Check(double const *const *parameters);

  Eigen::Vector2d CamProject(const Eigen::Vector3d &trans_xyz) const {
    Eigen::Vector2d proj;
    proj.x() = trans_xyz.x() / trans_xyz.z();
    proj.y() = trans_xyz.y() / trans_xyz.z();

    Eigen::Vector2d res;
    res.x() = proj.x() * fx + cx;
    res.y() = proj.y() * fy + cy;
    return res;
  }

  void SetInformation(const Eigen::Matrix2d &_information) {
    information = _information;
    Eigen::LLT<Eigen::Matrix2d> llt_of_information(information);
    sqrt_information = llt_of_information.matrixL().transpose();
  }

  void ComputeError(const double *const *parameters);
  double GetChi2() { return error_.dot(information * error_); };
  Eigen::Vector2d GetError() { return error_; }

  void SetFixed(bool fixed, const double *const *parameters) {
    fixed_ = fixed;
    if (parameters) ComputeError(parameters);
  }

  Eigen::Vector3d Xw;
  Eigen::Vector2d measurement;
  double fx, fy, cx, cy;
  Eigen::Matrix2d information = Eigen::Matrix2d::Identity();
  Eigen::Matrix2d sqrt_information = Eigen::Matrix2d::Identity();

 private:
  Eigen::Vector2d error_;
  bool fixed_ = false;
};

} // namespace dsl::relocalization

#endif // DSL_REPROJECTION_FUNCTOR_H
