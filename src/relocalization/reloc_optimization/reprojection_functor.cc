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

#include "relocalization/reloc_optimization/reprojection_functor.h"

namespace dsl::relocalization {

bool ReprojectionFunctor::Evaluate(const double *const *parameters, double *residuals, double **jacobians) const {
  Eigen::Map<Eigen::Vector2d> r(residuals);
  Eigen::Map<const Vec6> param0(parameters[0]);
  SE3 T_cw = SE3::exp(param0);
  Eigen::Vector3d Xc = T_cw * Xw;

  if (!fixed_) {
    r = sqrt_information * (measurement - CamProject(Xc));
  } else {
    r = sqrt_information * error_;
  }

  if (jacobians != nullptr) {
    if (jacobians[0] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> j(
          jacobians[0]);
      if (!fixed_) {
        double x = Xc.x();
        double y = Xc.y();
        double invz = 1.0 / Xc.z();
        double invz_2 = invz * invz;

        j(0, 0) = -invz * fx;
        j(0, 1) = 0;
        j(0, 2) = x * invz_2 * fx;
        j(0, 3) = x * y * invz_2 * fx;
        j(0, 4) = -(1 + (x * x * invz_2)) * fx;
        j(0, 5) = y * invz * fx;

        j(1, 0) = 0;
        j(1, 1) = -invz * fy;
        j(1, 2) = y * invz_2 * fy;
        j(1, 3) = (1 + y * y * invz_2) * fy;
        j(1, 4) = -x * y * invz_2 * fy;
        j(1, 5) = -x * invz * fy;

        j = sqrt_information * j;
      } else {
        j.setZero();
      }
    }
  }

  return true;
}

void ReprojectionFunctor::Check(const double *const *parameters) {
  Eigen::Map<const Vec6> param0(parameters[0]);
  SE3 T_cw = SE3::exp(param0);

  std::vector<double> res_vec(2);
  std::vector<double> jaco_vec(12);
  std::vector<double *> pjaco_vec(1);
  pjaco_vec[0] = jaco_vec.data();
  double *res = res_vec.data();
  double **jaco = pjaco_vec.data();

  Evaluate(parameters, res, jaco);
  std::cout << "check begins" << std::endl;
  std::cout << "analytical:" << std::endl;

  std::cout << Eigen::Map<Eigen::Matrix<double, 2, 1>>(res).transpose() << std::endl;
  std::cout << std::endl << Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>>(jaco[0]) << std::endl;

  ComputeError(parameters);
  Eigen::Vector2d residual = sqrt_information * error_;

  const double eps = 1e-6;
  Eigen::Matrix<double, 2, 7> num_jacobian;
  for (int k = 0; k < 6; k++) {

    int a = k / 3, b = k % 3;
    Vec6 delta = Vec6::Zero();
    delta[k] = eps;
    SE3 T_cw_plus = SE3::exp(delta) * T_cw;

    Eigen::Vector2d tmp_residual;

    Eigen::Vector3d Xc_plus = T_cw_plus * Xw;
    tmp_residual = sqrt_information * (measurement - CamProject(Xc_plus));

    num_jacobian.col(k) = (tmp_residual - residual) / eps;
  }
  std::cout << "numerical" << std::endl;
  std::cout << (sqrt_information * error_).transpose() << std::endl;
  std::cout << std::endl << num_jacobian.block<2, 6>(0, 0) << std::endl;
}

void ReprojectionFunctor::ComputeError(const double *const *parameters) {
  Eigen::Map<const Vec6> param0(parameters[0]);
  SE3 T_cw = SE3::exp(param0);
  Eigen::Vector3d Xc = T_cw * Xw;
  error_ = measurement - CamProject(Xc);
}
ReprojectionFunctor::ReprojectionFunctor(ReprojectionFunctor &functor) {
  Xw = functor.Xw;
  measurement = functor.measurement;
  fx = functor.fx;
  fy = functor.fy;
  cx = functor.cx;
  cy = functor.cy;
  information = functor.information;
  sqrt_information = functor.sqrt_information;

  error_ = functor.error_;
  fixed_ = functor.fixed_;
}

} // namespace dsl::relocalization