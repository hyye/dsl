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
// Created by hyye on 11/12/19.
//

#include "full_system/full_system.h"

namespace dsl {

void FullSystem::TraceNewCoarse(FrameHessian &fh) {
  // TODO: LINEARIZE_OPERATION

  int trace_total = 0, trace_good = 0, trace_oob = 0, trace_out = 0,
      trace_skip = 0, trace_badcondition = 0, trace_uninitialized = 0;

  Mat33f K = Mat33f::Identity();
  K(0, 0) = HCalib.fxl();
  K(1, 1) = HCalib.fyl();
  K(0, 2) = HCalib.cxl();
  K(1, 2) = HCalib.cyl();

  /// go through all active frames, exclude the current one fh
  for (std::unique_ptr<FrameHessian> &host : frame_hessians) {
    SE3 hostToNew = fh.PRE_world_to_cam * host->PRE_cam_to_world;
    Mat33f KRKi = K * hostToNew.rotationMatrix().cast<float>() * K.inverse();
    Mat33f R = hostToNew.rotationMatrix().cast<float>();
    Vec3f t = hostToNew.translation().cast<float>();

    Vec2f aff =
        AffLight::FromToVecExposure(host->exposure, fh.exposure,
                                    host->GetAffLight(), fh.GetAffLight())
            .cast<float>();

    for (std::unique_ptr<ImmaturePoint> &impt : host->immature_points) {
      impt->TraceOn(fh, KRKi, R, t, aff, HCalib);

      if (impt->last_trace_status == ImmaturePointStatus::IPS_GOOD)
        trace_good++;
      if (impt->last_trace_status == ImmaturePointStatus::IPS_BADCONDITION)
        trace_badcondition++;
      if (impt->last_trace_status == ImmaturePointStatus::IPS_OOB) trace_oob++;
      if (impt->last_trace_status == ImmaturePointStatus::IPS_OUTLIER)
        trace_out++;
      if (impt->last_trace_status == ImmaturePointStatus::IPS_SKIPPED)
        trace_skip++;
      if (impt->last_trace_status == ImmaturePointStatus::IPS_UNINITIALIZED)
        trace_uninitialized++;
      trace_total++;
    }
  }

  {
    char buff[1000];
    snprintf(buff, sizeof(buff),
             "ADD: TRACE: %d points. %d (%.0f%%) good. %d (%.0f%%) skip. %d "
             "(%.0f%%) badcond. %d (%.0f%%) oob. %d (%.0f%%) out. %d (%.0f%%) "
             "uninit.\n",
             trace_total, trace_good, 100 * trace_good / (float)trace_total,
             trace_skip, 100 * trace_skip / (float)trace_total,
             trace_badcondition, 100 * trace_badcondition / (float)trace_total,
             trace_oob, 100 * trace_oob / (float)trace_total, trace_out,
             100 * trace_out / (float)trace_total, trace_uninitialized,
             100 * trace_uninitialized / (float)trace_total);
    std::string buff_as_str = buff;
    DLOG(INFO) << buff_as_str;
  }
}

void FullSystem::MakeNewTraces(FrameHessian &new_frame) {
  int num_points_total = pixel_selector->MakeMaps(
      new_frame, selection_map, settingDesiredImmatureDensity);

  new_frame.point_hessians.reserve(num_points_total * 1.2f);
  new_frame.point_hessians_marginalized.reserve(num_points_total * 1.2f);
  new_frame.point_hessians_out.reserve(num_points_total * 1.2f);

  for (int y = patternPadding + 1; y < hG[0] - patternPadding - 2; y++)
    for (int x = patternPadding + 1; x < wG[0] - patternPadding - 2; x++) {
      int i = x + y * wG[0];
      if (selection_map[i] == 0) continue;
      if (!ValidArea(maskG[0], x, y)) continue;

      std::unique_ptr<ImmaturePoint> impt = std::make_unique<ImmaturePoint>(
          x, y, &new_frame, selection_map[i], HCalib);

      if (!std::isfinite(impt->energy_th)) {
        continue;
      }
      else {
        new_frame.immature_points.emplace_back(std::move(impt));
      }
    }
}

}