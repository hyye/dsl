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
// Created by hyye on 2/10/20.
//

#ifndef DSL_OPTIMIZATION_CALL_BACK_H_
#define DSL_OPTIMIZATION_CALL_BACK_H_

#include <ceres/ceres.h>
#include "full_system/full_system.h"

namespace dsl {

class OptimizationCallBack : public ceres::IterationCallback {
 public:
  OptimizationCallBack(FullSystem &full_system) : full_system_(full_system) {
    for (auto &&fh : full_system_.frame_hessians) {
      // poses_.push_back(fh->parameter_pose.GetEstimate());
      // abs_.push_back(fh->parameter_ab.GetEstimate());
    }
  }

  void Backup();
  bool CanBreak();

  virtual ceres::CallbackReturnType operator()(
      const ceres::IterationSummary &summary);

 private:
  FullSystem &full_system_;
  // std::vector<Vec6> poses_;
  // std::vector<Vec2> abs_;
};

}  // namespace dsl

#endif  // DSL_OPTIMIZATION_CALL_BACK_H_
