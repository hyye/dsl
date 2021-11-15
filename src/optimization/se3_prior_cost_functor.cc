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
// Created by hyye on 1/15/20.
//

#include "optimization/se3_prior_cost_functor.h"

namespace dsl {

SE3PriorCostFunctor::SE3PriorCostFunctor(const SE3 &measurement,
                               const information_t &information) {
  SetMeasurement(measurement);
  SetInformation(information);
}

void SE3PriorCostFunctor::SetMeasurement(const SE3 &measurement) {
  measurement_ = measurement;
  measurement_inv_ = measurement.inverse();
}

void SE3PriorCostFunctor::SetInformation(const information_t &information) {
  information_ = information;
  Eigen::LLT<information_t> llt_of_information(information_);
  sqrt_information_ = llt_of_information.matrixL().transpose();
}

bool SE3PriorCostFunctor::Evaluate(const double *const *parameters,
                              double *residuals, double **jacobians) const {
  Eigen::Map<const Vec6> param0(parameters[0]);
  SE3 estimation = SE3::exp(param0);
  SE3 error = measurement_inv_ * estimation;
  Eigen::Map<Eigen::Matrix<double, 6, 1> > weighted_error(residuals);
  weighted_error = sqrt_information_ * error.log();

  if (jacobians != nullptr) {
    if (jacobians[0] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor> > J0(
          jacobians[0]);
      J0 = sqrt_information_ * measurement_inv_.Adj();
    }
  }
  return true;
}

}  // namespace dsl