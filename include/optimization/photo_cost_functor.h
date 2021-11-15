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
// Created by hyye on 11/18/19.
//

#ifndef DSL_PHOTO_COST_FUNCTOR_H_
#define DSL_PHOTO_COST_FUNCTOR_H_

#include <ceres/ceres.h>
#include "dsl_common.h"
#include "util/num_type.h"

namespace dsl {

class PhotoCostFunctor : public ceres::SizedCostFunction<8, 6, 2, 6, 2, 1> {
 public:
  PhotoCostFunctor(PointFrameResidual* residual, CalibHessian& HCalib,
                  bool baseline = false);

  Vec3f* dI;

  virtual bool Evaluate(double const* const* parameters, double* residuals,
                        double** jacobians) const;

  void SetEmpty(double* residuals, double** jacobians) const;

  void SetResidualFromNew(PointFrameResidual *pfr) const;
  void SetOOB(double* residuals, double** jacobians, PointFrameResidual *pfr) const;

  double CalcHdd(const SE3f& host_to_target, double idist);

 // protected:
  PointHessian* point_;
  FrameHessian* host_;
  FrameHessian* target_;
  PointFrameResidual* pfr_;
  CalibHessian& HCalib_;

  mutable bool first_eval = true;

  bool baseline_;
  // mutable bool idist_converged_ = false;
};

}  // namespace dsl

#endif  // DSL_PHOTO_COST_FUNCTOR_H_
