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
// Created by hyye on 11/13/19.
//

#ifndef DSL_RESIDUAL_H_
#define DSL_RESIDUAL_H_

#include <ceres/ceres.h>
#include "dsl_common.h"
#include "optimization/parameter_map.h"

namespace dsl {

struct PointFrameResidual {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ResState res_state;
  ResState new_res_state;
  double res_energy;
  double new_res_energy;
  double new_res_energy_with_outiler;

  double res_raw;
  double new_res_raw;
  static int instanceCounter;

  bool is_new;

  void ResetOOB();
  double Linearize(CalibHessian &HCalib);

  void SetState(ResState s) { res_state = s; }

  void ApplyRes(bool copy_jacobians);

  EfResidual *ef_residual;
  PointHessian *point;
  FrameHessian *host;
  FrameHessian *target;

  std::unique_ptr<ceres::CostFunction> cost_function;
  std::unique_ptr<ResidualBlockSpec> res_blk_spec =
      std::make_unique<ResidualBlockSpec>();

  Vec2f projected_to[MAX_RES_PER_POINT];
  Vec3f center_projected_to;
  Vec3f center_projected_to_backup;
  float ph_idist = -1, ph_idist_backup;

  Vec4f plane_coeff;
  bool idist_converged = false;
  bool valid_plane = false;

  Eigen::Matrix<double, 8, 1> Jd;
  Eigen::Matrix<double, 8, 1> Jd_backup;

  PointFrameResidual(PointHessian *_point, FrameHessian *_host,
                     FrameHessian *_target, CalibHessian *_HCalib = 0);

  ~PointFrameResidual() { --instanceCounter; }

  void SetResPlane();
};

}  // namespace dsl

#endif  // DSL_RESIDUAL_H_
