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
// Created by hyye on 11/5/19.
//

#ifndef DSL_EF_STRUCT_H_
#define DSL_EF_STRUCT_H_

#include "dsl_common.h"

namespace dsl {

struct EfFrame {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  FrameHessian *frame;
  std::vector<std::unique_ptr<EfPoint>> ef_points;
  EfFrame(FrameHessian *fh) : frame(fh) {}

  int idx;
  int frame_id;
};

enum class EfPointStatus { GOOD = 0, MARGINALIZE, DROP };

struct EfPoint {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  PointHessian *point;
  EfFrame *host;

  float HdiF;
  EfPointStatus state_flag;
  int idx_in_points;

  std::vector<std::unique_ptr<EfResidual>> ef_residuals;
  EfPoint(PointHessian *ph, EfFrame *ef_frame) : point(ph), host(ef_frame) {
    state_flag = EfPointStatus::GOOD;
  }
};

struct EfResidual {
  PointFrameResidual *pfr;
  EfPoint *point;
  EfFrame *host;
  EfFrame *target;

  int idx_in_all;

  int host_idx;
  int target_idx;

  bool is_linearized;

  bool is_active_and_is_good_new;  // TODO: change the name
  inline const bool &IsActive() const { return is_active_and_is_good_new; }

  void FixLinearizationF(EnergyFunction *ef);
  void TakeDataF();

  EfResidual(PointFrameResidual *_pfr, EfPoint *_point, EfFrame *_host,
             EfFrame *_target)
      : pfr(_pfr), point(_point), host(_host), target(_target) {
    is_linearized = false;
    is_active_and_is_good_new = false;
  }
};

class EnergyFunction {
 public:
  EnergyFunction();
  std::vector<std::unique_ptr<EfFrame>> ef_frames;
  // ownership not change
  void InsertFrame(FrameHessian *fh, CalibHessian &HCalib);
  void InsertPoint(std::unique_ptr<PointHessian> &ph);
  void InsertResidual(PointFrameResidual* r);

  void DropPointsF();
  void RemovePoint(EfPoint *efp);
  void DropResidual(EfResidual *efr);

  void MarginalizeFrame(EfFrame *eff);

  void MarginalizePointsF();

  void MakeIdx();
  void SetAdjointsF(CalibHessian &HCalib);

  bool ef_adjoints_valid = false;
  bool ef_indices_valid = false;

  int num_frames, num_points, num_residuals;

  std::vector<EfPoint *> all_points;
  std::vector<EfPoint *> all_points_to_marg;

  std::vector<Mat88> ad_host;
  std::vector<Mat88> ad_target;

  std::vector<Mat88f> ad_hostf;
  std::vector<Mat88f> ad_targetf;
};

}  // namespace dsl

#endif  // DSL_EF_STRUCT_H_
