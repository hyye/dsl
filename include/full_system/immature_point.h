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
// Created by hyye on 11/6/19.
//

#ifndef DSL_IMMATURE_POINT_H_
#define DSL_IMMATURE_POINT_H_

#include "dsl_common.h"
#include "util/num_type.h"

namespace dsl {

struct ImmaturePointTemporaryResidual {
 public:
  ResState res_state;
  double res_energy;
  ResState new_res_state;
  double new_res_energy;
  FrameHessian* target;
};

enum class ImmaturePointStatus {
  IPS_GOOD = 0,      // traced well and good
  IPS_OOB,           // OOB: end tracking & marginalize!
  IPS_OUTLIER,       // energy too high: if happens again: outlier!
  IPS_SKIPPED,       // traced well and good (but not actually traced).
  IPS_BADCONDITION,  // not traced because of bad condition.
  IPS_UNINITIALIZED
};  // not even traced once.

class ImmaturePoint {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ImmaturePointStatus last_trace_status;
  Vec2f last_trace_uv;
  float last_trace_pixel_interval;

  // static values
  float color[MAX_RES_PER_POINT];
  float weights[MAX_RES_PER_POINT];

  Mat22f gradH;
  Vec2f gradH_ev;
  Mat22f gradH_eig;
  float energy_th;
  float u, v;
  // on the unit sphere
  float x, y, z;
  FrameHessian* host;
  int idx_in_immature_points;

  /// non-max suppression
  float quality;

  float my_type;

  float idist_min;
  float idist_max;

  ImmaturePoint(int _u, int _v, FrameHessian* _host, float type,
                CalibHessian& HCalib);
  double LinearizeResidual(CalibHessian& HCalib, const float outlier_th_slack,
                           ImmaturePointTemporaryResidual& tmp_res, float& Hdd,
                           float& bd, float idist);
  ImmaturePointStatus TraceOn(FrameHessian& frame,
                              const Mat33f& host_to_frame_KRKi,
                              const Mat33f& host_to_frame_R,
                              const Vec3f& host_to_frame_t,
                              const Vec2f& host_to_frame_aff,
                              CalibHessian& HCalib);
};

}  // namespace dsl

#endif  // DSL_IMMATURE_POINT_H_
