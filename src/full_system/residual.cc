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

#include "full_system/residual.h"
#include "full_system/hessian_blocks.h"
#include <full_system/ef_struct.h>
#include "optimization/homo_cost_functor.h"

namespace dsl {

void PointFrameResidual::SetResPlane() {
  if (point->valid_plane) {
    valid_plane = true;
    plane_coeff = point->plane_coeff;
  }
}

int PointFrameResidual::instanceCounter = 0;

PointFrameResidual::PointFrameResidual(PointHessian* _point,
                                       FrameHessian* _host,
                                       FrameHessian* _target,
                                       CalibHessian* _HCalib)
    : point(_point), host(_host), target(_target) {
  ef_residual = nullptr;
  ResetOOB();
  ++instanceCounter;

  is_new = true;

  SetResPlane();

  cost_function = std::make_unique<HomoCostFunctor>(this, *_HCalib, settingBaseline);
}

void PointFrameResidual::ResetOOB() {
  new_res_energy = res_energy = 0;
  new_res_raw = res_raw = 0;
  new_res_state = ResState::OUTLIER;

  SetState(ResState::IN);
}

double PointFrameResidual::Linearize(dsl::CalibHessian& HCalib) {
  // FIXME: Linearize
}

void PointFrameResidual::ApplyRes(bool copy_jacobians) {
  if (copy_jacobians) {
    if (res_state == ResState::OOB) {
      assert(!ef_residual->is_active_and_is_good_new);
      return;  // can never go back from OOB
    }
    if (new_res_state == ResState::IN) {
      ef_residual->is_active_and_is_good_new = true;
      ef_residual->TakeDataF();
    } else {
      ef_residual->is_active_and_is_good_new = false;
    }
  }

  SetState(new_res_state);
  res_energy = new_res_energy;
  res_raw = new_res_raw;
}

}  // namespace dsl