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

std::unique_ptr<PointHessian> FullSystem::OptimizeImmaturePoint(
    ImmaturePoint &point, int min_obs,
    std::vector<ImmaturePointTemporaryResidual> &residuals) {
  int nres = 0;
  /// initialize res from all the other fhs, host <--> the other frames
  for (std::unique_ptr<FrameHessian> &fh : frame_hessians) {
    if (fh.get() != point.host) {
      residuals[nres].new_res_energy = residuals[nres].res_energy = 0;
      residuals[nres].new_res_state = ResState::OUTLIER;
      residuals[nres].res_state = ResState::IN;
      residuals[nres].target = fh.get();
      nres++;
    }
  }
  assert(nres == ((int)frame_hessians.size()) - 1);

  float last_energy = 0;
  float lastHdd = 0;
  float lastbd = 0;
  float current_idist = (point.idist_max + point.idist_min) * 0.5f;

  for (int i = 0; i < nres; i++) {
    last_energy += point.LinearizeResidual(HCalib, 1000, residuals[i], lastHdd,
                                           lastbd, current_idist);
    residuals[i].res_state = residuals[i].new_res_state;
    residuals[i].res_energy = residuals[i].new_res_energy;
  }

  if (!std::isfinite(last_energy) || lastHdd < settingMinIdistHAct) {
    return nullptr;
  }

  float lambda = 0.1;
  for (int iteration = 0; iteration < settingGNItsOnPointActivation;
       iteration++) {
    float H = lastHdd;
    H *= 1 + lambda;
    float step = (1.0 / H) * lastbd;
    float newIdepth = current_idist - step;

    float newHdd = 0;
    float newbd = 0;
    float new_energy = 0;
    for (int i = 0; i < nres; i++)
      new_energy += point.LinearizeResidual(HCalib, 1, residuals[i], newHdd,
                                            newbd, newIdepth);

    if (!std::isfinite(last_energy) || newHdd < settingMinIdistHAct) {
      return nullptr;
    }

    if (new_energy < last_energy) {
      current_idist = newIdepth;
      lastHdd = newHdd;
      lastbd = newbd;
      last_energy = new_energy;
      for (int i = 0; i < nres; i++) {
        residuals[i].res_state = residuals[i].new_res_state;
        residuals[i].res_energy = residuals[i].new_res_energy;
      }

      lambda *= 0.5;
    } else {
      lambda *= 5;
    }

    if (fabsf(step) < 0.0001 * current_idist) break;
  }

  if (!std::isfinite(current_idist)) {
    printf("MAJOR ERROR! point idistance is nan after initialization (%f).\n",
           current_idist);
    return nullptr;
  }

  int num_good_res = 0;
  for (int i = 0; i < nres; i++)
    if (residuals[i].res_state == ResState::IN) num_good_res++;

  /// if less than minimum observed (1 in default case), return it as -1
  if (num_good_res < min_obs) {
    return nullptr;  // yeah I'm like 99% sure this is OK on 32bit systems.
  }

  // NOTE: create new PointHessian
  // TODO: HomoResidual
  //  PointHessian *p = new HomoPointHessian(point, HCalib);
  std::unique_ptr<PointHessian> p =
      std::make_unique<PointHessian>(point, HCalib);
  if (!std::isfinite(p->energy_th)) {
    return nullptr;
  }

  /// lastResiduals reset
  p->last_residuals[0].first = 0;
  p->last_residuals[0].second = ResState::OOB;
  p->last_residuals[1].first = 0;
  p->last_residuals[1].second = ResState::OOB;
  p->SetIdistZero(current_idist);
  p->SetIdist(current_idist);
  p->SetPhStatus(PointHessian::ACTIVE);

  for (int i = 0; i < nres; i++) {
    if (residuals[i].res_state == ResState::IN) {
      // TODO: HomoResidual
      // PointFrameResidual *r = new HomoResidual(p, p->host,
      // residuals[i].target, settingBaseline);
      std::unique_ptr<PointFrameResidual> r =
          std::make_unique<PointFrameResidual>(p.get(), p->host,
                                               residuals[i].target, &HCalib);
      PointFrameResidual *r_ptr = r.get();
      r_ptr->new_res_energy = r->res_energy = 0;
      r_ptr->new_res_raw = r->res_raw = 0;
      r_ptr->new_res_state = ResState::OUTLIER;
      r_ptr->SetState(ResState::IN);
      p->residuals.emplace_back(std::move(r));

      if (r_ptr->target == frame_hessians.back().get()) {
        /// for new keyframe
        p->last_residuals[0].first = r_ptr;
        p->last_residuals[0].second = ResState::IN;
      } else if (r_ptr->target ==
          (frame_hessians.size() < 2
           ? nullptr
           : frame_hessians[frame_hessians.size() - 2].get())) {
        /// not the new keyframe, host to last [0] is not set here
        p->last_residuals[1].first = r_ptr;
        p->last_residuals[1].second = ResState::IN;
      }
    }
  }

  return p;
}

void FullSystem::ActivatePointsMTReductor(
    std::vector<std::unique_ptr<PointHessian>> *optimized,
    std::vector<std::unique_ptr<ImmaturePoint>> *to_optimize, int min, int max,
    Eigen::Matrix<double, 10, 1> &stats, int tid) {
  std::vector<ImmaturePointTemporaryResidual> tr(frame_hessians.size());
  for (int k = min; k < max; k++) {
    /// GN optimization
    (*optimized)[k] = OptimizeImmaturePoint(*(*to_optimize)[k], 1, tr);
  }
}

void FullSystem::ActivatePointsMT() {
  if (ef->num_points < settingDesiredPointDensity * 0.66)
    current_min_act_dist -= 0.8;
  if (ef->num_points < settingDesiredPointDensity * 0.8)
    current_min_act_dist -= 0.5;
  else if (ef->num_points < settingDesiredPointDensity * 0.9)
    current_min_act_dist -= 0.2;
  else if (ef->num_points < settingDesiredPointDensity)
    current_min_act_dist -= 0.1;

  if (ef->num_points > settingDesiredPointDensity * 1.5)
    current_min_act_dist += 0.8;
  if (ef->num_points > settingDesiredPointDensity * 1.3)
    current_min_act_dist += 0.5;
  if (ef->num_points > settingDesiredPointDensity * 1.15)
    current_min_act_dist += 0.2;
  if (ef->num_points > settingDesiredPointDensity) current_min_act_dist += 0.1;

  if (current_min_act_dist < 0) current_min_act_dist = 0;
  if (current_min_act_dist > 4) current_min_act_dist = 4;

  if (!settingDebugoutRunquiet) {
    char buff[100];
    snprintf(buff, sizeof(buff),
             "SPARSITY:  MinActDist %f (need %d points, have %d points)!\n",
             current_min_act_dist, (int)(settingDesiredPointDensity),
             ef->num_points);
    std::string buff_as_str = buff;
    LOG(INFO) << buff_as_str;
  }

  std::unique_ptr<FrameHessian> &newest_hs = frame_hessians.back();

  timing::Timer mk_timer("dsl/keyframe/activate/mk");
  coarse_distance_map->MakeK(HCalib);
  coarse_distance_map->MakeDistanceMap(frame_hessians, *newest_hs);
  mk_timer.Stop();

  std::vector<std::unique_ptr<ImmaturePoint>> to_optimize;
  to_optimize.reserve(20000);

  int total_impt = 0;

  // go through all active frames
  for (std::unique_ptr<FrameHessian> &host : frame_hessians) {
    if (host == newest_hs) continue;

    SE3 fhToNew = newest_hs->PRE_world_to_cam * host->PRE_cam_to_world;
    Mat33f KRKi =
        (coarse_distance_map->K[1] * fhToNew.rotationMatrix().cast<float>() *
         coarse_distance_map->Ki[0]);
    Vec3f Kt =
        (coarse_distance_map->K[1] * fhToNew.translation().cast<float>());
    Mat33f R = fhToNew.rotationMatrix().cast<float>();
    Vec3f t = fhToNew.translation().cast<float>();

    for (unsigned int i = 0; i < host->immature_points.size(); i += 1) {
      std::unique_ptr<ImmaturePoint> &impt = host->immature_points[i];
      impt->idx_in_immature_points = i;

      /// NOTE: check if after traceNewCoarse the points are valid
      // delete points that have never been traced successfully, or that are
      // outlier on the last trace.
      if (!std::isfinite(impt->idist_max) ||
          impt->last_trace_status == ImmaturePointStatus::IPS_OUTLIER) {
        // remove point.
        host->immature_points[i] = nullptr;
        continue;
      }

      // can activate only if this is true.
      bool can_activate =
          (impt->last_trace_status == ImmaturePointStatus::IPS_GOOD ||
           impt->last_trace_status == ImmaturePointStatus::IPS_SKIPPED ||
           impt->last_trace_status == ImmaturePointStatus::IPS_BADCONDITION ||
           impt->last_trace_status == ImmaturePointStatus::IPS_OOB) &&
          impt->last_trace_pixel_interval < 8 &&
          impt->quality > settingMinTraceQuality &&
          (impt->idist_max + impt->idist_min) > 0;

      ++total_impt;

      // if I cannot activate the point, skip it. Maybe also delete it.
      if (!can_activate) {
        if (impt->last_trace_status == ImmaturePointStatus::IPS_GOOD ||
            impt->last_trace_status == ImmaturePointStatus::IPS_SKIPPED) {
        }
        // if point will be out afterwards, delete it instead.
        if (impt->host->flagged_for_marginalization ||
            impt->last_trace_status == ImmaturePointStatus::IPS_OOB) {
          host->immature_points[i] = nullptr;
        }
        continue;
      }

      /// OOB points will not be added in to toOptimize
      // see if we need to activate point due to distance map.
      Vec3f ptp =
          SpaceToPlane(R * Vec3f(impt->x, impt->y, impt->z) +
                           t * 0.5f * (impt->idist_max + impt->idist_min),
                       coarse_distance_map->fx[1], coarse_distance_map->fy[1],
                       coarse_distance_map->cx[1], coarse_distance_map->cy[1],
                       coarse_distance_map->xi);
      int u = ptp[0] / ptp[2] + 0.5f;
      int v = ptp[1] / ptp[2] + 0.5f;

      if ((u > 0 && v > 0 && u < wG[1] && v < hG[1] &&
           ValidArea(maskG[1], u, v))) {
        /// NOTE: make sure the points to be activated not too close to the
        /// existed points.
        float dist = coarse_distance_map->fwd_warped_dist_final[u + wG[1] * v] +
                     (ptp[0] - floorf((float)(ptp[0])));

        /// NOTE: my_type is block size from select
        if (dist >= current_min_act_dist * impt->my_type) {
          coarse_distance_map->AddIntoDistFinal(u, v);
          to_optimize.emplace_back(std::move(impt));
        }
      } else {
        host->immature_points[i] = nullptr;
      }
    }
  }

  LOG(INFO) << "ACTIVATE:" << (int)to_optimize.size()
            << " total_impt: " << total_impt;

  // new phs are created in ActivatePoints
  std::vector<std::unique_ptr<PointHessian>> optimized;
  optimized.resize(to_optimize.size());

  timing::Timer optpoint_timer("dsl/keyframe/activate/opt");
  if (settingMultiThreading) {
    thread_reduce.Reduce(
        std::bind(&FullSystem::ActivatePointsMTReductor, this, &optimized,
                  &to_optimize, std::placeholders::_1, std::placeholders::_2,
                  std::placeholders::_3, std::placeholders::_4),
        0, to_optimize.size(), 50);
  } else {
    Vec10 dummy_vec10;
    ActivatePointsMTReductor(&optimized, &to_optimize, 0, to_optimize.size(),
                             dummy_vec10, 0);
  }
  optpoint_timer.Stop();

  timing::Timer insert_timer("dsl/keyframe/activate/insert");
  for (unsigned k = 0; k < to_optimize.size(); k++) {
    // optimized new point
    std::unique_ptr<PointHessian> &newpoint = optimized[k];
    PointHessian *newpoint_ptr = newpoint.get();
    std::unique_ptr<ImmaturePoint> &impt = to_optimize[k];

    if (newpoint_ptr != nullptr &&
        newpoint_ptr != (PointHessian *)((long)(-1))) {
      newpoint_ptr->host->immature_points[impt->idx_in_immature_points] =
          nullptr;

      // NOTE: push_back optimized point
      // TODO: HomoResidual
      if (!newpoint_ptr->host->vertex_map.empty() && !newpoint_ptr->host->normal_map.empty()) {
        // LOG(INFO) << "@#@#@# activate homo";
        newpoint_ptr->SetPlane(newpoint_ptr->host->vertex_map,
                               newpoint_ptr->host->normal_map);
        for (unsigned int i = 0; i < newpoint_ptr->residuals.size(); i++) {
          newpoint_ptr->residuals[i]->SetResPlane();
        }
      } else {
        LOG(ERROR) << "@#@#@# no vn info!!!";
      }
      
      ef->InsertPoint(newpoint);

      /// insert new PointFrameResidual
      for (std::unique_ptr<PointFrameResidual> &r : newpoint_ptr->residuals) {
        ef->InsertResidual(r.get());
      }
      assert(newpoint_ptr->ef_point != nullptr);

    } else if (newpoint_ptr == (PointHessian *)((long)(-1)) ||
               impt->last_trace_status == ImmaturePointStatus::IPS_OOB) {
      impt->host->immature_points[impt->idx_in_immature_points] = nullptr;
    } else {
      assert(newpoint_ptr == nullptr ||
             newpoint_ptr == (PointHessian *)((long)(-1)));
    }
  }
  insert_timer.Stop();

  /// pop out the immaturePoints which are set to 0 (deleted) before
  timing::Timer pop_timer("dsl/keyframe/activate/pop");
  for (std::unique_ptr<FrameHessian> &host : frame_hessians) {
    for (int i = 0; i < (int)host->immature_points.size(); i++) {
      if (host->immature_points[i] == nullptr) {
        std::swap(host->immature_points[i], host->immature_points.back());
        host->immature_points.pop_back();
        --i;
      }
    }
  }
  pop_timer.Stop();
}

}