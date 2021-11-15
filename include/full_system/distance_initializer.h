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
// Created by hyye on 11/11/19.
//

#ifndef DSL_DISTANCE_INITIALIER_H_
#define DSL_DISTANCE_INITIALIER_H_

#include "hessian_blocks.h"
#include "pixel_selector.h"
#include "util/global_calib.h"
#include "util/global_funcs.h"
#include "util/num_type.h"

namespace dsl {

struct Pnt {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  float u, v;

  // idistance / isgood / energy during optimization.
  float idist;
  bool is_good;
  Vec2f energy;  // (UenergyPhotometric, energyRegularizer)
  bool is_good_new;
  float idist_new;
  Vec2f energy_new;

  float iR;
  float iR_sum_num;

  float last_hessian;
  float last_hessian_new;

  // idx (x+y*w) of closest point one pyramid level above.
  int parent;
  float parent_dist;

  // idx (x+y*w) of up to 10 nearest points in pixel space.
  int neighbours[10];
  float neighbours_dist[10];

  float my_type;
  float outlier_th;
};

class DistanceInitializer {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  DistanceInitializer(int w, int h);
  ~DistanceInitializer() {}

  void MakeK(CalibHessian &HCalib);

  void MakeNN();

  void SetFirstDistance(CalibHessian &HCalib,
                        std::unique_ptr<FrameHessian> &new_fh,
                        const std::vector<float> &dist_metric,
                        const SE3 &pose_in_world);

  bool TrackFrameDepth(FrameHessian &new_fh,
                       /*std::vector<IOWrap::Output3DWrapper *> &wraps,*/
                       const SE3 &pose_in_world);

  int frame_id;
  std::array<std::vector<Pnt>, PYR_LEVELS> points;

  AffLight this_to_next_aff;
  SE3 this_to_next;

  std::unique_ptr<FrameHessian> first_frame;
  FrameHessian *new_frame;

  SE3 first_to_world;

  Mat33 K[PYR_LEVELS];
  Mat33 Ki[PYR_LEVELS];
  double fx[PYR_LEVELS];
  double fy[PYR_LEVELS];
  double fxi[PYR_LEVELS];
  double fyi[PYR_LEVELS];
  double cx[PYR_LEVELS];
  double cy[PYR_LEVELS];
  double cxi[PYR_LEVELS];
  double cyi[PYR_LEVELS];
  int w[PYR_LEVELS];
  int h[PYR_LEVELS];
  double xi;

  bool snapped;
  int snapped_at;
};

struct FLANNPointcloud {
  inline FLANNPointcloud() {
    num = 0;
    points = 0;
  }
  inline FLANNPointcloud(int n, Pnt *p) : num(n), points(p) {}
  int num;
  Pnt *points;
  inline size_t kdtree_get_point_count() const { return num; }
  inline float kdtree_distance(const float *p1, const size_t idx_p2,
                               size_t /*size*/) const {
    const float d0 = p1[0] - points[idx_p2].u;
    const float d1 = p1[1] - points[idx_p2].v;
    return d0 * d0 + d1 * d1;
  }

  inline float kdtree_get_pt(const size_t idx, int dim) const {
    if (dim == 0)
      return points[idx].u;
    else
      return points[idx].v;
  }
  template <class BBOX>
  bool kdtree_get_bbox(BBOX & /* bb */) const {
    return false;
  }
};

}  // namespace dsl

#endif  // DSL_DISTANCE_INITIALIER_H_
