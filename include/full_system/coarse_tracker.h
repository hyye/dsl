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
// Created by hyye on 11/7/19.
//

// Coarse to fine tracking, adapted from DSO

#ifndef DSL_COARSE_TRACKER_H_
#define DSL_COARSE_TRACKER_H_

#include "coarse_distance_map.h"
#include "dsl_common.h"
#include "hessian_blocks.h"
#include "optimization/matrix_accumulators.h"

namespace dsl {

class CoarseTracker {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  CoarseTracker(int _w, int _h);
  ~CoarseTracker();

  void MakeK(CalibHessian &HCalib);

  void MakeCoarseDistL0(
      std::vector<std::unique_ptr<FrameHessian>> &frame_hessians);

  void SetCoarseTrackingRef(
      std::vector<std::unique_ptr<FrameHessian>> &frame_hessians);

  bool TrackNewestCoarse(FrameHessian &new_fh, SE3 &last_to_new_out,
                         AffLight &aff_light_out, int coarsest_lvl,
                         const Vec5 &min_res_for_abort
                         /*,IOWrap::Output3DWrapper *wrap = 0*/);

  Vec6 CalcRes(int lvl, const SE3 &ref_to_new, AffLight aff_light,
               float cutoff_th);
  void CalcGSSSE(int lvl, Mat88 &H_out, Vec8 &b_out, const SE3 &ref_to_new,
                 AffLight aff_light);

  Mat33f K[PYR_LEVELS];
  Mat33f Ki[PYR_LEVELS];
  float fx[PYR_LEVELS];
  float fy[PYR_LEVELS];
  float fxi[PYR_LEVELS];
  float fyi[PYR_LEVELS];
  float cx[PYR_LEVELS];
  float cy[PYR_LEVELS];
  float cxi[PYR_LEVELS];
  float cyi[PYR_LEVELS];
  float xi;
  int w[PYR_LEVELS];
  int h[PYR_LEVELS];

  FrameHessian *last_ref_fh;
  AffLight last_ref_aff_light;
  FrameHessian *new_frame;
  int ref_frame_id;

  Vec5 last_residuals;
  Vec3 last_flow_indicators;
  double first_coarse_rmse;

  std::array<std::vector<float>, PYR_LEVELS> idist;
  std::array<std::vector<float>, PYR_LEVELS> weight_sums;
  std::array<std::vector<float>, PYR_LEVELS> weight_sums_bak;

  std::array<std::vector<float>, PYR_LEVELS> pc_u;
  std::array<std::vector<float>, PYR_LEVELS> pc_v;
  std::array<std::vector<float>, PYR_LEVELS> pc_x;
  std::array<std::vector<float>, PYR_LEVELS> pc_y;
  std::array<std::vector<float>, PYR_LEVELS> pc_z;
  std::array<std::vector<float>, PYR_LEVELS> pc_idist;
  std::array<std::vector<float>, PYR_LEVELS> pc_color;
  int pc_n[PYR_LEVELS];

  // warped buffers
  std::vector<float> buf_warped_idist;
  std::vector<float> buf_warped_u;
  std::vector<float> buf_warped_v;
  std::vector<float> buf_warped_x;
  std::vector<float> buf_warped_y;
  std::vector<float> buf_warped_z;
  std::vector<float> buf_warped_idepth_xi;
  std::vector<float> buf_warped_dx;
  std::vector<float> buf_warped_dy;
  std::vector<float> buf_warped_residual;
  std::vector<float> buf_warped_weight;
  std::vector<float> buf_warped_ref_color;
  int buf_warped_n;

  Accumulator9 acc;
};

}  // namespace dsl

#endif  // DSL_COARSE_TRACKER_H_
