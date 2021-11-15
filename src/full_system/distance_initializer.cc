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

#include "full_system/distance_initializer.h"
#include "nanoflann.hpp"

namespace dsl {

DistanceInitializer::DistanceInitializer(int w, int h) { frame_id = -1; }

void DistanceInitializer::MakeK(dsl::CalibHessian& HCalib) {
  w[0] = wG[0];
  h[0] = hG[0];

  fx[0] = HCalib.fxl();
  fy[0] = HCalib.fyl();
  cx[0] = HCalib.cxl();
  cy[0] = HCalib.cyl();
  xi = HCalib.xil();

  // NOTE: originally, cx should be in some place like 3.5
  for (int level = 1; level < pyrLevelsUsed; ++level) {
    w[level] = w[0] >> level;
    h[level] = h[0] >> level;
    fx[level] = fx[level - 1] * 0.5;
    fy[level] = fy[level - 1] * 0.5;
    cx[level] = (cx[0] + 0.5) / ((int)1 << level) - 0.5;
    cy[level] = (cy[0] + 0.5) / ((int)1 << level) - 0.5;
  }

  for (int level = 0; level < pyrLevelsUsed; ++level) {
    K[level] << fx[level], 0.0, cx[level], 0.0, fy[level], cy[level], 0.0, 0.0,
        1.0;
    Ki[level] = K[level].inverse();
    fxi[level] = Ki[level](0, 0);
    fyi[level] = Ki[level](1, 1);
    cxi[level] = Ki[level](0, 2);
    cyi[level] = Ki[level](1, 2);
  }
}

// NOTE: NN in points & parents in the upper layer
void DistanceInitializer::MakeNN() {
  const float NNDistFactor = 0.05;

  typedef nanoflann::KDTreeSingleIndexAdaptor<
      nanoflann::L2_Simple_Adaptor<float, FLANNPointcloud>, FLANNPointcloud, 2>
      KDTree;

  // build indices
  FLANNPointcloud pcs[PYR_LEVELS];
  std::array<std::unique_ptr<KDTree>, PYR_LEVELS> indices;
  for (int i = 0; i < pyrLevelsUsed; i++) {
    pcs[i] = FLANNPointcloud(points[i].size(), points[i].data());
    indices[i] = std::make_unique<KDTree>(
        2, pcs[i], nanoflann::KDTreeSingleIndexAdaptorParams(5));
    indices[i]->buildIndex();
  }

  const int nn = 10;

  // find NN & parents
  for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
    std::vector<Pnt>& pts = points[lvl];
    int npts = points[lvl].size();

    int ret_index[nn];
    float ret_dist[nn];
    nanoflann::KNNResultSet<float, int, int> result_set(nn);
    nanoflann::KNNResultSet<float, int, int> result_set1(1);

    for (int i = 0; i < npts; i++) {
      result_set.init(ret_index, ret_dist);
      Vec2f pt = Vec2f(pts[i].u, pts[i].v);
      indices[lvl]->findNeighbors(result_set, (float*)&pt,
                                  nanoflann::SearchParams());
      int myidx = 0;
      float sum_df = 0;
      for (int k = 0; k < nn; k++) {
        pts[i].neighbours[myidx] = ret_index[k];
        float df = expf(-ret_dist[k] * NNDistFactor);
        sum_df += df;
        pts[i].neighbours_dist[myidx] = df;
        assert(ret_index[k] >= 0 && ret_index[k] < npts);
        myidx++;
      }
      for (int k = 0; k < nn; k++) pts[i].neighbours_dist[k] *= 10 / sum_df;

      if (lvl < pyrLevelsUsed - 1) {
        result_set1.init(ret_index, ret_dist);
        pt = pt * 0.5f - Vec2f(0.25f, 0.25f);
        indices[lvl + 1]->findNeighbors(result_set1, (float*)&pt,
                                        nanoflann::SearchParams());

        pts[i].parent = ret_index[0];
        pts[i].parent_dist = expf(-ret_dist[0] * NNDistFactor);

        assert(ret_index[0] >= 0 && ret_index[0] < points[lvl + 1].size());
      } else {
        pts[i].parent = -1;
        pts[i].parent_dist = -1;
      }
    }
  }

  // done.
}

void DistanceInitializer::SetFirstDistance(
    CalibHessian& HCalib, std::unique_ptr<FrameHessian>& new_fh,
    const std::vector<float>& dist_metric, const dsl::SE3& pose_in_world) {
  MakeK(HCalib);
  first_frame = std::move(new_fh);

  first_to_world = pose_in_world;

  PixelSelector sel(w[0], h[0]);

  // NOTE: PixelSelectorStatus map
  std::vector<float> status_map(w[0] * h[0]);

  float densities[] = {0.03, 0.05, 0.15, 0.5, 1};
  int lvl = 0;

  sel.current_potential = 3;
  int npts;
  if (lvl == 0)
    npts = sel.MakeMaps(*first_frame, status_map, densities[lvl] * w[0] * h[0],
                        1, 2);

  int wl = w[lvl], hl = h[lvl];
  points[lvl].clear();

  npts = 0;
  for (int y = patternPadding + 1; y < hl - patternPadding - 2; y++)
    for (int x = patternPadding + 1; x < wl - patternPadding - 2; x++) {
      if ((lvl == 0 && status_map[x + y * wl] != 0)) {
        ++npts;
      }
    }

  points[lvl].reserve(npts);

  std::vector<Pnt>& pl = points[lvl];
  for (int y = patternPadding + 1; y < hl - patternPadding - 2; y++) {
    for (int x = patternPadding + 1; x < wl - patternPadding - 2; x++) {
      if ((lvl == 0 && status_map[x + y * wl] != 0)) {
        Pnt new_pnt;
        new_pnt.u = x + 0.1;
        new_pnt.v = y + 0.1;

        float idist = 0;
        if (dist_metric[x + y * wl] != 0) {
          idist = 1 / dist_metric[x + y * wl];
        } else {
          continue;
        }

        new_pnt.idist = idist;
        new_pnt.iR = idist;
        new_pnt.is_good = true;
        new_pnt.energy.setZero();
        new_pnt.last_hessian = 0;
        new_pnt.last_hessian_new = 0;
        new_pnt.my_type = status_map[x + y * wl];

        Eigen::Vector3f* cpt = first_frame->dIp[lvl].data() + x + y * w[lvl];
        float sumGrad2 = 0;
        for (int idx = 0; idx < patternNum; idx++) {
          int dx = patternP[idx][0];
          int dy = patternP[idx][1];
          float absgrad = cpt[dx + dy * w[lvl]].tail<2>().squaredNorm();
          sumGrad2 += absgrad;
        }

        new_pnt.outlier_th = patternNum * settingOutlierTh;
        pl.emplace_back(new_pnt);

        assert(pl.size() <= npts);
      }
    }
  }

  // MakeNN();

  this_to_next = SE3();
  snapped = false;
  frame_id = snapped_at = 0;
}

// WARNING: thsi function and the second pose_in_world is not necessary
bool DistanceInitializer::TrackFrameDepth(FrameHessian& new_fh,
                                          const dsl::SE3& pose_in_world) {
  static int track_frame_depth_count = -1;
  LOG(INFO) << "track_frame_depth_count: " << ++track_frame_depth_count;

  new_frame = &new_fh;

  SE3 first_to_new = pose_in_world.inverse() * first_to_world;
  SE3 ref_to_new_current = first_to_new;
  // NOTE: aff contains a, b parameters
  AffLight ref_to_new_aff_current = this_to_next_aff;
  if (first_frame->exposure > 0 && new_frame->exposure > 0) {
    ref_to_new_aff_current =
        AffLight(logf(new_frame->exposure / first_frame->exposure),
                 0);  // coarse approximation.
  }
  snapped = true;
  this_to_next = ref_to_new_current;
  this_to_next_aff = ref_to_new_aff_current;

  frame_id++;
  if (!snapped) {
    snapped_at = 0;
  }
  LOG(WARNING) << "trackFrameDepth: " << frame_id;

  return snapped && frame_id > snapped_at;
}

}