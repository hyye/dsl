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
// Created by hyye on 8/25/20.
//

#include "relocalization/reloc_optimization/surfel_reprojection_functor.h"

namespace dsl::relocalization {

SurfelReprojectionFunctor::SurfelReprojectionFunctor(const Eigen::Vector3d &_x0,
                                                     const Eigen::Vector3d &_x1,
                                                     const Eigen::Vector3d &_n_w,
                                                     double _d_w) :
    x0_bar(_x0), x1_bar(_x1), n_w(_n_w), d_w(_d_w) {
}
bool SurfelReprojectionFunctor::Evaluate(const double *const *parameters,
                                         double *residuals,
                                         double **jacobians) const {
  Eigen::Map<Eigen::Vector2d> r(residuals);
  Eigen::Map<const Vec6> param0(parameters[0]);
  Eigen::Map<const Vec6> param1(parameters[1]);
  SE3 T_wc0 = SE3::exp(param0);
  SE3 T_wci = SE3::exp(param1);
  SE3 T_ciw = T_wci.inverse();
  double rho = -n_w.dot(T_wc0.rotationMatrix() * x0_bar) / (n_w.dot(T_wc0.translation()) + d_w);
  Eigen::Vector3d xw = T_wc0 * (1 / rho * x0_bar);
  Eigen::Vector3d x1_star = T_ciw * xw;
  r = (x1_star / x1_star.z() - x1_bar).head<2>();

  if (jacobians != nullptr) {
    Eigen::Matrix<double, 2, 3> proj_mat;
    proj_mat << 1 / x1_star.z(), 0, -x1_star.x() / (x1_star.z() * x1_star.z()),
        0, 1 / x1_star.z(), -x1_star.y() / (x1_star.z() * x1_star.z());
    if (jacobians[0] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> j(jacobians[0]);
      Eigen::Matrix<double, 3, 6> dxwdxi;
      Eigen::Matrix3d R_wc0 = T_wc0.rotationMatrix();
      Eigen::Vector3d t_wc0 = T_wc0.translation();
      Eigen::Vector3d xw_bar = R_wc0 * x0_bar;
      double n_w_t_xw_bar = n_w.dot(xw_bar);
      double n_w_t_t_wc0_p_d_w = n_w.dot(t_wc0) + d_w;

      dxwdxi.block(0, 0, 3, 3) = -xw_bar * n_w.transpose() / (xw_bar.dot(n_w)) + Eigen::Matrix3d::Identity();
      dxwdxi.block(0, 3, 3, 3) =
          (Eigen::Matrix3d::Identity() * (n_w_t_t_wc0_p_d_w) / n_w_t_xw_bar
              - xw_bar * (n_w_t_t_wc0_p_d_w) / (n_w_t_xw_bar * n_w_t_xw_bar) * n_w.transpose())
              * SO3::hat(xw_bar)
              + xw_bar * (n_w.transpose() * SO3::hat(t_wc0)) / n_w_t_xw_bar
              - SO3::hat(t_wc0);
      j = proj_mat * T_ciw.rotationMatrix() * dxwdxi;
    }
    if (jacobians[1] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> j(jacobians[1]);
      Eigen::Matrix<double, 3, 6> dxdxi;
      dxdxi.block(0, 0, 3, 3) = -T_ciw.rotationMatrix();
      dxdxi.block(0, 3, 3, 3) = T_ciw.rotationMatrix() * SO3::hat(xw);
      j = proj_mat * dxdxi;
    }
  }
  return true;
}

void SurfelReprojectionFunctor::Check(const double *const *parameters) {
  Eigen::Map<const Vec6> param0(parameters[0]);
  Eigen::Map<const Vec6> param1(parameters[1]);
  SE3 T_wc0 = SE3::exp(param0);
  SE3 T_wci = SE3::exp(param1);

  std::vector<double> res_vec(2);
  std::vector<double> jaco_vec0(12);
  std::vector<double> jaco_vec1(12);
  std::vector<double *> pjaco_vec(2);
  pjaco_vec[0] = jaco_vec0.data();
  pjaco_vec[1] = jaco_vec1.data();
  double *res = res_vec.data();
  double **jaco = pjaco_vec.data();

  Evaluate(parameters, res, jaco);
  std::cout << "check begins" << std::endl;
  std::cout << "analytical:" << std::endl;

  std::cout << Eigen::Map<Eigen::Matrix<double, 2, 1>>(res).transpose() << std::endl;
  std::cout << std::endl << Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>>(jaco[0]) << std::endl;
  std::cout << std::endl << Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>>(jaco[1]) << std::endl;

  std::cout << std::endl << "numerical:" << std::endl;
  const double eps = 1e-6;
  Eigen::Matrix<double, 2, 6> num_jacobian;
  Eigen::Vector2d residual = Eigen::Map<Eigen::Matrix<double, 2, 1>>(res);
  for (int k = 0; k < 6; k++) {
    Vec6 delta = Vec6::Zero();
    delta[k] = eps;
    SE3 T_wc0_plus = SE3::exp(delta) * T_wc0;

    SE3 T_ciw = T_wci.inverse();
    double rho = -n_w.dot(T_wc0_plus.rotationMatrix() * x0_bar) / (n_w.dot(T_wc0_plus.translation()) + d_w);
    Eigen::Vector3d xw = T_wc0_plus * (1 / rho * x0_bar);
    Eigen::Vector3d x1_star = T_ciw * xw;
    Eigen::Vector2d tmp_residual = (x1_star / x1_star.z() - x1_bar).head<2>();

    num_jacobian.col(k) = (tmp_residual - residual) / eps;
  }
  std::cout << std::endl << num_jacobian.block<2, 6>(0, 0) << std::endl;

  for (int k = 0; k < 6; k++) {
    Vec6 delta = Vec6::Zero();
    delta[k] = eps;
    SE3 T_wci_plus = SE3::exp(delta) * T_wci;

    SE3 T_ciw_plus = T_wci_plus.inverse();
    double rho = -n_w.dot(T_wc0.rotationMatrix() * x0_bar) / (n_w.dot(T_wc0.translation()) + d_w);
    Eigen::Vector3d xw = T_wc0 * (1 / rho * x0_bar);
    Eigen::Vector3d x1_star = T_ciw_plus * xw;
    Eigen::Vector2d tmp_residual = (x1_star / x1_star.z() - x1_bar).head<2>();

    num_jacobian.col(k) = (tmp_residual - residual) / eps;
  }
  std::cout << std::endl << num_jacobian.block<2, 6>(0, 0) << std::endl;
}

void SurfelReprojectionFunctor::ComputeError(const double *const *parameters) {
  Eigen::Map<const Vec6> param0(parameters[0]);
  Eigen::Map<const Vec6> param1(parameters[1]);
  // LOG(INFO) << param0 << std::endl << param1;
  SE3 T_wc0 = SE3::exp(param0);
  SE3 T_wci = SE3::exp(param1);

  std::vector<double> res_vec(2);
  double *res = res_vec.data();

  Evaluate(parameters, res, NULL);

  error = sqrt(res_vec[0] * res_vec[0] + res_vec[1] * res_vec[1]);
}

}