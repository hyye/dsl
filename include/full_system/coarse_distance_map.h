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
// Created by hyye on 11/8/19.
//

#ifndef DSL_COARSE_DISTANCE_MAP_H_
#define DSL_COARSE_DISTANCE_MAP_H_

#include "dsl_common.h"
#include "hessian_blocks.h"

namespace dsl {

class CoarseDistanceMap {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  CoarseDistanceMap(int _w, int _h);
  ~CoarseDistanceMap() {}

  void MakeDistanceMap(
      std::vector<std::unique_ptr<FrameHessian>> &frame_hessians,
      FrameHessian &frame);

  // TODO: is it necessary to have a copy here?
  void MakeK(CalibHessian &HCalib);

  void AddIntoDistFinal(int u, int v);

  void GrowDistBFS(/*int num_bfs1*/);

  std::vector<float> fwd_warped_dist_final;
  std::vector<Eigen::Vector2i> bfs_list1;
  std::vector<Eigen::Vector2i> bfs_list2;

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
};


}

#endif  // DSL_COARSE_DISTANCE_MAP_H_
