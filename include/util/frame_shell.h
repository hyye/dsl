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

#ifndef DSL_FRAME_SHELL_H_
#define DSL_FRAME_SHELL_H_

#include "num_type.h"

namespace dsl {

struct FrameShell {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  int id;
  int incoming_id;
  double timestamp;

  // set once after tracking
  SE3 cam_to_ref;
  FrameShell* tracking_ref;

  SE3 cam_to_world;
  AffLight aff_light;
  bool pose_valid;

  // statistics
  int statistics_outlier_res;
  int statistics_good_res;
  int marginalized_at;
  double moved_by_opt;

  FrameShell() {
    id = 0;
    pose_valid = true;
    cam_to_world = SE3();
    timestamp = 0;
    marginalized_at = -1;
    moved_by_opt = statistics_outlier_res=statistics_good_res=0;;
    tracking_ref=0;
    cam_to_ref = SE3();
  }

};

}

#endif // DSL_FRAME_SHELL_H_
