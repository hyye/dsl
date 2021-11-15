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

#include "full_system/hessian_blocks.h"

namespace dsl {

void FrameFramePrecalc::Set(dsl::FrameHessian *host, dsl::FrameHessian *target,
                            dsl::CalibHessian &HCalib) {
  this->host = host;
  this->target = target;

  /// This value is not updated during the optimization, but after
  SE3 left_to_left_0 =
      target->GetWorldToCamEvalPT() * host->GetWorldToCamEvalPT().inverse();
  PRE_RTll_0 = (left_to_left_0.rotationMatrix()).cast<float>();
  PRE_tTll_0 = (left_to_left_0.translation()).cast<float>();

  /// This value will update during the optimization
  SE3 left_to_left = target->PRE_world_to_cam * host->PRE_cam_to_world;
  PRE_RTll = (left_to_left.rotationMatrix()).cast<float>();
  PRE_tTll = (left_to_left.translation()).cast<float>();
  distance_ll = left_to_left.translation().norm();

  Mat33f K = Mat33f::Zero();
  K(0, 0) = HCalib.fxl();
  K(1, 1) = HCalib.fyl();
  K(0, 2) = HCalib.cxl();
  K(1, 2) = HCalib.cyl();
  K(2, 2) = 1;
  PRE_KRKiTll = K * PRE_RTll * K.inverse();
  PRE_RKiTll = PRE_RTll * K.inverse();
  PRE_KtTll = K * PRE_tTll;

  PRE_aff_mode =
      AffLight::FromToVecExposure(host->exposure, target->exposure,
                                  host->GetAffLight(), target->GetAffLight())
          .cast<float>();
  PRE_b0_mode = host->GetAffLightZero().b;
}

FrameHessian::FrameHessian() {
  ++instanceCounter;
  flagged_for_marginalization = false;
  frame_id = -1;
  ef_frame = nullptr;
  frame_energy_th = 8 * 8 * patternNum;
  state.setZero();
  state_zero.setZero();
}

void FrameHessian::MakeImages(float *color, CalibHessian *HCalib) {
  for (int i = 0; i < pyrLevelsUsed; i++) {
    dIp[i] = std::move(std::vector<Vec3f>(wG[i] * hG[i]));
    abs_sq_grad[i] = std::move(std::vector<float>(wG[i] * hG[i]));
  }
  dI = dIp[0].data();

  // make d0
  int w = wG[0];
  int h = hG[0];
  for (int i = 0; i < w * h; ++i) {
    dI[i][0] = color[i];
  }

  // NOTE: level l, each pixel in dIp is Vector3f: intensity, dx, dy
  for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
    int wl = wG[lvl], hl = hG[lvl];
    Vec3f *dI_l = dIp[lvl].data();

    float *dabs_l = abs_sq_grad[lvl].data();
    if (lvl > 0) {
      int lvlm1 = lvl - 1;
      int wlm1 = wG[lvlm1];
      Vec3f *dI_lm = dIp[lvlm1].data();

      // NOTE: averaging from l-1
      for (int y = 0; y < hl; y++)
        for (int x = 0; x < wl; x++) {
          dI_l[x + y * wl][0] =
              0.25f * (dI_lm[2 * x + 2 * y * wlm1][0] +
                       dI_lm[2 * x + 1 + 2 * y * wlm1][0] +
                       dI_lm[2 * x + 2 * y * wlm1 + wlm1][0] +
                       dI_lm[2 * x + 1 + 2 * y * wlm1 + wlm1][0]);
        }
    }

    for (int idx = wl; idx < wl * (hl - 1); idx++) {
      float dx = 0.5f * (dI_l[idx + 1][0] - dI_l[idx - 1][0]);
      float dy = 0.5f * (dI_l[idx + wl][0] - dI_l[idx - wl][0]);

      if (!std::isfinite(dx)) dx = 0;
      if (!std::isfinite(dy)) dy = 0;

      dI_l[idx][1] = dx;
      dI_l[idx][2] = dy;

      dabs_l[idx] = dx * dx + dy * dy;

      // NOTE: no B and Binv,HCalib not used
    }
  }
}

PointHessian::PointHessian(const ImmaturePoint &raw_point,
                           CalibHessian &HCalib) {
  ++instanceCounter;
  host = raw_point.host;
  has_dist_prior = false;

  idist_hessian = 0;
  max_rel_baseline = 0;
  num_good_residuals = 0;

  u = raw_point.u;
  v = raw_point.v;
  p_sphere = Vec3f(raw_point.x, raw_point.y, raw_point.z);
  assert(std::isfinite(raw_point.idist_max));

  my_type = raw_point.my_type;

  SetIdist((raw_point.idist_max + raw_point.idist_min) * 0.5);
  SetPhStatus(PointHessian::INACTIVE);

  std::memcpy(color, raw_point.color, sizeof(float) * MAX_RES_PER_POINT);
  std::memcpy(weights, raw_point.weights, sizeof(float) * MAX_RES_PER_POINT);

  energy_th = raw_point.energy_th;
  ef_point = nullptr;
}

bool PointHessian::SetPlane(
    std::vector<Vec4f, Eigen::aligned_allocator<Vec4f>> &vertex_map,
    std::vector<Vec4f, Eigen::aligned_allocator<Vec4f>> &normal_map) {
  if (!vertex_map.empty() && !normal_map.empty()) {
    int index = u + v * wG[0];
    const Eigen::Vector3f &vertex = vertex_map[index].head<3>();
    const Eigen::Vector3f &normal = normal_map[index].head<3>();
    if (IsValid(vertex) && IsValid(normal)) {
      float d = -normal.dot(vertex);
      plane_coeff = Vec4f(normal.x(), normal.y(), normal.z(), d);
      valid_plane = true;
      return true;
    }
  }
  return false;
}

}  // namespace dsl