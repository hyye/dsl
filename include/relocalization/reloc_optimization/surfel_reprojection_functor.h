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

#ifndef DSL_SURFEL_REPROJECTION_FUNCTOR_H
#define DSL_SURFEL_REPROJECTION_FUNCTOR_H

#include <ceres/ceres.h>
#include "dsl_common.h"

namespace dsl::relocalization {

/// \brief on unit image plane
class SurfelReprojectionFunctor : public ceres::SizedCostFunction<2, 6, 6> {
 public:
  SurfelReprojectionFunctor(const Eigen::Vector3d &_x0,
                            const Eigen::Vector3d &_x1,
                            const Eigen::Vector3d &_n_w,
                            double _d_w);
  virtual bool Evaluate(double const *const *parameters, double *residuals,
                        double **jacobians) const;
  void Check(double const *const *parameters);

  void ComputeError(const double *const *parameters);

  Eigen::Vector3d x0_bar, x1_bar;
  Eigen::Vector3d n_w;
  double d_w;
  double error;
};

} // namespace dsl::relocalization

#endif // DSL_SURFEL_REPROJECTION_FUNCTOR_H
