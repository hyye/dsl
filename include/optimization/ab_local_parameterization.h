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
// Created by hyye on 1/18/20.
//

#ifndef DSL_AB_LOCAL_PARAMETERIZATION_H_
#define DSL_AB_LOCAL_PARAMETERIZATION_H_

#include "util/num_type.h"
#include "dsl_common.h"
#include <ceres/ceres.h>

namespace dsl {

struct AbLocalParameterization : public ceres::LocalParameterization {
  virtual bool Plus(const double* x, const double* delta,
                    double* x_plus_delta) const {
    Eigen::Map<const Vec2> tangent(x);

    Eigen::Map<const Vec2> d_tangent(delta);

    Eigen::Map<Vec2> tangent_plus(x_plus_delta);

    Vec2 scaled_dx = d_tangent;
    if (!settigNoScalingAtOpt) {
      scaled_dx.x() *= SCALE_A;
      scaled_dx.y() *= SCALE_B;
    }

    tangent_plus = tangent + scaled_dx;

    return true;
  }

  virtual bool ComputeJacobian(const double* x, double* jacobian) const {
    Eigen::Map<Eigen::Matrix<double, 2, 2, Eigen::RowMajor>> j(jacobian);
    j.setIdentity();
    return true;
  }

  virtual int GlobalSize() const { return 2; };
  virtual int LocalSize() const { return 2; };
};

}

#endif  // DSL_AB_LOCAL_PARAMETERIZATION_H_
