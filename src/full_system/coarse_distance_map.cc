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

#include "full_system/coarse_distance_map.h"

namespace dsl {

/**
 * CoarseDistanceMap
 * @param _w
 * @param _h
 */
CoarseDistanceMap::CoarseDistanceMap(int _w, int _h) {
  fwd_warped_dist_final = std::move(std::vector<float>(_w * _h / 4));
  bfs_list1.reserve(_w * _h / 4);
  bfs_list2.reserve(_w * _h / 4);

  w[0] = h[0] = 0;
}

/**
 * make distance map from the projection points from previous (nearest dist to
 * the existed points)
 * @param frame_hessians all fh with the new key frame
 * @param frame the new key frame
 */
void CoarseDistanceMap::MakeDistanceMap(
    std::vector<std::unique_ptr<FrameHessian>>& frame_hessians,
    dsl::FrameHessian& frame) {
  int w1 = w[1];
  int h1 = h[1];
  int wh1 = w1 * h1;
  for (int i = 0; i < wh1; ++i) {
    fwd_warped_dist_final[i] = 1000;
  }

  bfs_list1.clear();

  for (std::unique_ptr<FrameHessian>& fh : frame_hessians) {
    if (&frame == fh.get()) continue;

    SE3 fh_to_new = frame.PRE_world_to_cam * fh->PRE_cam_to_world;
    const Mat33f R = fh_to_new.rotationMatrix().cast<float>();
    const Vec3f t = fh_to_new.translation().cast<float>();

    for (std::unique_ptr<PointHessian>& ph : fh->point_hessians) {
      Vec3f ptp = SpaceToPlane(R * ph->p_sphere + t * ph->idist, fx[1], fy[1],
                               cx[1], cy[1], xi);
      // round to new fh coordinate
      int u = ptp[0] / ptp[2] + 0.5f;
      int v = ptp[1] / ptp[2] + 0.5f;
      if (!(u > 0 && v > 0 && u < w[1] && v < h[1])) continue;
      fwd_warped_dist_final[u + w1 * v] = 0;
      bfs_list1.emplace_back(u, v);
    }

    GrowDistBFS();
  }
}

void CoarseDistanceMap::MakeK(CalibHessian& HCalib) {
  w[0] = wG[0];
  h[0] = hG[0];

  fx[0] = HCalib.fxl();
  fy[0] = HCalib.fyl();
  cx[0] = HCalib.cxl();
  cy[0] = HCalib.cyl();
  xi = HCalib.xil();

  /// subpixel
  for (int level = 1; level < pyrLevelsUsed; ++level) {
    w[level] = w[0] >> level;
    h[level] = h[0] >> level;
    fx[level] = fx[level - 1] * 0.5;
    fy[level] = fy[level - 1] * 0.5;
    cx[level] = (cx[0] + 0.5) / ((int)1 << level) - 0.5;
    cy[level] = (cy[0] + 0.5) / ((int)1 << level) - 0.5;
  }

  /// inverse
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

void CoarseDistanceMap::AddIntoDistFinal(int u, int v) {
  assert(w[0] != 0);
  fwd_warped_dist_final[u + w[1] * v] = 0;
  bfs_list1.emplace_back(u, v);
  GrowDistBFS();
}

void CoarseDistanceMap::GrowDistBFS() {
  assert(w[0] != 0);
  int w1 = w[1], h1 = h[1];
  for (int k = 1; k < 40; k++) {
    int num_bfs2 = bfs_list1.size();
    std::swap(bfs_list1, bfs_list2);
    bfs_list1.clear();

    if (k % 2 == 0) {
      for (int i = 0; i < bfs_list2.size(); ++i) {
        int x = bfs_list2[i][0];
        int y = bfs_list2[i][1];
        if (x == 0 || y == 0 || x == w1 - 1 || y == h1 - 1) continue;
        int idx = x + y * w1;

        if (fwd_warped_dist_final[idx + 1] > k) {
          fwd_warped_dist_final[idx + 1] = k;
          bfs_list1.emplace_back(x + 1, y);
        }
        if (fwd_warped_dist_final[idx - 1] > k) {
          fwd_warped_dist_final[idx - 1] = k;
          bfs_list1.emplace_back(x - 1, y);
        }
        if (fwd_warped_dist_final[idx + w1] > k) {
          fwd_warped_dist_final[idx + w1] = k;
          bfs_list1.emplace_back(x, y + 1);
        }
        if (fwd_warped_dist_final[idx - w1] > k) {
          fwd_warped_dist_final[idx - w1] = k;
          bfs_list1.emplace_back(x, y - 1);
        }
      }
    } else {
      for (int i = 0; i < num_bfs2; i++) {
        int x = bfs_list2[i][0];
        int y = bfs_list2[i][1];
        if (x == 0 || y == 0 || x == w1 - 1 || y == h1 - 1) continue;
        int idx = x + y * w1;

        if (fwd_warped_dist_final[idx + 1] > k) {
          fwd_warped_dist_final[idx + 1] = k;
          bfs_list1.emplace_back(x + 1, y);
        }
        if (fwd_warped_dist_final[idx - 1] > k) {
          fwd_warped_dist_final[idx - 1] = k;
          bfs_list1.emplace_back(x - 1, y);
        }
        if (fwd_warped_dist_final[idx + w1] > k) {
          fwd_warped_dist_final[idx + w1] = k;
          bfs_list1.emplace_back(x, y + 1);
        }
        if (fwd_warped_dist_final[idx - w1] > k) {
          fwd_warped_dist_final[idx - w1] = k;
          bfs_list1.emplace_back(x, y - 1);
        }

        if (fwd_warped_dist_final[idx + 1 + w1] > k) {
          fwd_warped_dist_final[idx + 1 + w1] = k;
          bfs_list1.emplace_back(x + 1, y + 1);
        }
        if (fwd_warped_dist_final[idx - 1 + w1] > k) {
          fwd_warped_dist_final[idx - 1 + w1] = k;
          bfs_list1.emplace_back(x - 1, y + 1);
        }
        if (fwd_warped_dist_final[idx - 1 - w1] > k) {
          fwd_warped_dist_final[idx - 1 - w1] = k;
          bfs_list1.emplace_back(x - 1, y - 1);
        }
        if (fwd_warped_dist_final[idx + 1 - w1] > k) {
          fwd_warped_dist_final[idx + 1 - w1] = k;
          bfs_list1.emplace_back(x + 1, y - 1);
        }
      }
    }
  }

  bfs_list1.clear();  // clear before exit
}

}