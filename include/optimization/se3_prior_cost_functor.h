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

#ifndef DSL_SE3_COST_FUNCTOR_H_
#define DSL_SE3_COST_FUNCTOR_H_

#include <ceres/ceres.h>
#include "dsl_common.h"
#include "util/num_type.h"

namespace dsl {

class SE3PriorCostFunctor : public ceres::SizedCostFunction<6, 6> {
 public:
  typedef Eigen::Matrix<double, 6, 6> information_t;
  SE3PriorCostFunctor(const SE3& measurement, const information_t& information);

  void SetMeasurement(const SE3& measurement);
  const SE3& GetMeasurement() { return measurement_; }

  void SetInformation(const information_t& information);
  const information_t& GetInformation() { return information_; }

  virtual bool Evaluate(double const* const* parameters, double* residuals,
                        double** jacobians) const;

 private:
  SE3 measurement_;
  SE3 measurement_inv_;
  information_t information_;
  information_t sqrt_information_;
};

}  // namespace dsl

#endif  // DSL_SE3_COST_FUNCTOR_H_
