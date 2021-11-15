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

#include "full_system/full_system.h"
#include "optimization/ab_local_parameterization.h"
#include "optimization/se3_local_parameterization.h"

namespace dsl {

int FrameHessian::instanceCounter = 0;
int PointHessian::instanceCounter = 0;
int CalibHessian::instanceCounter = 0;

FullSystem::FullSystem() {
  selection_map.resize(wG[0] * hG[0]);

  coarse_distance_map = std::make_unique<CoarseDistanceMap>(wG[0], hG[0]);
  coarse_tracker = std::make_unique<CoarseTracker>(wG[0], hG[0]);
  coarse_tracker_for_new_kf = std::make_unique<CoarseTracker>(wG[0], hG[0]);
  distance_initializer = std::make_unique<DistanceInitializer>(wG[0], hG[0]);
  pixel_selector = std::make_unique<PixelSelector>(wG[0], hG[0]);

  last_coarse_rmse.setConstant(100);

  current_min_act_dist = 2;
  initialized = false;
  is_lost = false;
  init_failed = false;

  need_new_kf_after = -1;

  linearize_operation = true;
  run_mapping = true;

  ef = std::make_unique<EnergyFunction>();
  marginalization_info_ = std::make_unique<MarginalizationInfo>();

  parameter_map = std::make_unique<ParameterMap>();
  se3_local_parameterization = std::make_unique<SE3LocalParameterization>();
  ab_local_parameterization = std::make_unique<AbLocalParameterization>();
}

void FullSystem::AddActiveFrame(ImageAndExposure &image, int id,
                                const std::vector<float> &dist_metric,
                                const dsl::SE3 &pose_in_world,
                                Vec4f *vertex_map, Vec4f *normal_map) {
  if (is_lost) return;

  // TODO: LINEARIZE_OPERATION

  std::unique_ptr<FrameHessian> fh = std::make_unique<FrameHessian>();
  std::unique_ptr<FrameShell> shell = std::make_unique<FrameShell>();

  shell->cam_to_world = SE3();
  shell->aff_light = AffLight(0, 0);
  shell->marginalized_at = shell->id = all_frame_shells.size();
  shell->timestamp = image.timestamp;
  shell->incoming_id = id;
  fh->shell = shell.get();

  if (distance_initializer->frame_id < 0) {
    // WARNING: noise not added
    shell->cam_to_world = pose_in_world;
  } else {
    // TODO: remove this later? this value is not used
    shell->cam_to_world = distance_initializer->first_to_world;
  }
  all_frame_shells.emplace_back(std::move(shell));
  fh->exposure = image.exposure_time;
  fh->MakeImages(image.image.data(), &HCalib);

  if (!initialized) {
    if (distance_initializer->frame_id < 0) {
      // fh ownership changed
      FrameHessian *fh_ptr = fh.get();
      distance_initializer->SetFirstDistance(HCalib, fh, dist_metric,
                                             pose_in_world);
      InitializeFromInitializer(*fh_ptr, vertex_map, normal_map);
    } else {
      fh->shell->pose_valid = false;
    }
    return;
  } else {
    if (coarse_tracker_for_new_kf->ref_frame_id >
        coarse_tracker->ref_frame_id) {
      // TODO: LINEARIZE_OPERATION
      std::swap(coarse_tracker, coarse_tracker_for_new_kf);
    }

    timing::Timer tracker_timer("dsl/tracking");
    Vec4 tres = TrackNewCoarse(*fh);
    tracker_timer.Stop();

    if (!std::isfinite((double)tres[0]) || !std::isfinite((double)tres[1]) ||
        !std::isfinite((double)tres[2]) || !std::isfinite((double)tres[3])) {
      LOG(INFO) << "Initial Tracking failed: LOST!";
      is_lost = true;
      return;
    }

    bool need_to_make_kf;

    if (settingKeyframesPerSecond > 0) {
      need_to_make_kf =
          all_frame_shells.size() == 1 ||
          (fh->shell->timestamp - all_frame_shells.back()->timestamp) >
              0.95f / settingKeyframesPerSecond;
    } else {
      Vec2 ref_to_fh = AffLight::FromToVecExposure(
          coarse_tracker->last_ref_fh->exposure, fh->exposure,
          coarse_tracker->last_ref_aff_light, fh->shell->aff_light);
      DLOG(INFO) << "need to make kf? t:"
                 << settingKfGlobalWeight * settingMaxShiftWeightT *
                        sqrtf((double)tres[1]) / (wG[0] + hG[0])
                 << ", 0:"
                 << settingKfGlobalWeight * settingMaxShiftWeightR *
                        sqrtf((double)tres[2]) / (wG[0] + hG[0])
                 << ", rt:"
                 << settingKfGlobalWeight * settingMaxShiftWeightRT *
                        sqrtf((double)tres[3]) / (wG[0] + hG[0])
                 << ", aff:"
                 << settingKfGlobalWeight * settingMaxAffineWeight *
                        fabs(logf((float)ref_to_fh[0]));

      /// FLOW CHECK and BRIGHTNESS CHECK
      need_to_make_kf = all_frame_shells.size() == 1 ||
                        settingKfGlobalWeight * settingMaxShiftWeightT *
                                    sqrtf((double)tres[1]) / (wG[0] + hG[0]) +
                                settingKfGlobalWeight * settingMaxShiftWeightR *
                                    sqrtf((double)tres[2]) / (wG[0] + hG[0]) +
                                settingKfGlobalWeight *
                                    settingMaxShiftWeightRT *
                                    sqrtf((double)tres[3]) / (wG[0] + hG[0]) +
                                settingKfGlobalWeight * settingMaxAffineWeight *
                                    fabs(logf((float)ref_to_fh[0])) >
                            1 ||
                        2 * coarse_tracker->first_coarse_rmse < tres[0];
    }

    if (need_to_make_kf) {
      // TODO: simplify this
      if (vertex_map && normal_map) {
        int w = wG[0];
        int h = hG[0];
        if (frame_hessians.back()) {
          FrameHessian *last_fh = frame_hessians.back().get();
          if (last_fh->vertex_map.empty() && last_fh->normal_map.empty()) {
            // LOG(WARNING) << "@#@#@# SET TO FH";
            last_fh->vertex_map.resize(w * h);
            last_fh->normal_map.resize(w * h);
            for (int i = 0; i < w * h; i++) {
              last_fh->vertex_map[i] = vertex_map[i];
              last_fh->normal_map[i] = normal_map[i];
            }
          } else {
            LOG(WARNING) << "already set";
          }
        }
      }
    }

    DLOG(WARNING) << "THIS IS " << (need_to_make_kf ? "KF" : "NON-KF");
    DeliverTrackedFrame(fh, need_to_make_kf);
    is_keyframe = need_to_make_kf;
    return;
  }
}

void FullSystem::InitializeFromInitializer(FrameHessian &new_frame,
                                           Vec4f *vertex_map,
                                           Vec4f *normal_map) {
  // TODO: LINEARIZE_OPERATION
  FrameHessian *first_frame = distance_initializer->first_frame.get();

  first_frame->idx = frame_hessians.size();
  first_frame->frame_id = all_kf_shells.size();
  ef->InsertFrame(first_frame, HCalib);
  all_kf_shells.push_back(first_frame->shell);
  frame_hessians.emplace_back(std::move(distance_initializer->first_frame));

  if (vertex_map && normal_map) {
    int w = wG[0];
    int h = hG[0];
    if (frame_hessians.back()) {
      FrameHessian *last_fh = frame_hessians.back().get();
      if (last_fh->vertex_map.empty() && last_fh->normal_map.empty()) {
        last_fh->vertex_map.resize(w * h);
        last_fh->normal_map.resize(w * h);
        for (int i = 0; i < w * h; i++) {
          last_fh->vertex_map[i] = vertex_map[i];
          last_fh->normal_map[i] = normal_map[i];
        }
      } else {
        LOG(WARNING) << "already set";
      }
    }
  }

  SetPreCalcValues();

  first_frame->point_hessians.reserve(wG[0] * hG[0] * 0.2f);
  first_frame->point_hessians_marginalized.reserve(wG[0] * hG[0] * 0.2f);
  first_frame->point_hessians_out.reserve(wG[0] * hG[0] * 0.2f);

  float keep_percentage =
      settingDesiredPointDensity / distance_initializer->points[0].size();

  if (!settingDebugoutRunquiet) {
    char buff[100];
    snprintf(buff, sizeof(buff),
             "Initialization: keep %.1f%% (need %d, have %d)!\n",
             100 * keep_percentage, (int)(settingDesiredPointDensity),
             (int)distance_initializer->points[0].size());
    std::string buff_as_str = buff;
    LOG(INFO) << buff_as_str;
  }

  for (int i = 0; i < distance_initializer->points[0].size(); ++i) {
    if (rand() / (float)RAND_MAX > keep_percentage) continue;
    const Pnt &point = distance_initializer->points[0][i];
    ImmaturePoint pt(point.u + 0.5f, point.v + 0.5f, first_frame, point.my_type,
                     HCalib);
    if (!std::isfinite(pt.energy_th)) {
      continue;
    }

    pt.idist_max = point.idist * 1.2;
    pt.idist_min = point.idist / 1.2;

    std::unique_ptr<PointHessian> ph =
        std::make_unique<PointHessian>(pt, HCalib);
    ph->convereged_ph_idist = true;

    if (!std::isfinite(ph->energy_th)) {
      continue;
    }

    ph->SetIdist(point.iR);
    ph->SetIdistZero(ph->idist);
    ph->has_dist_prior = true;
    ph->SetPhStatus(PointHessian::ACTIVE);

    ph->SetPlane(ph->host->vertex_map, ph->host->normal_map);
    // LOG(INFO) << (ph->valid_plane ? "yyy" : "nnn");

    ef->InsertPoint(ph);
  }

  SE3 first_to_new = distance_initializer->this_to_next;

  {
    // TODO: LINEARIZE_OPERATION
    first_frame->shell->aff_light = AffLight(0, 0);
    first_frame->SetZeroEvalPT(first_frame->shell->cam_to_world.inverse(),
                               first_frame->shell->aff_light);
    first_frame->shell->tracking_ref = nullptr;
    first_frame->shell->cam_to_ref = SE3();

    new_frame.shell->cam_to_world =
        first_frame->shell->cam_to_world * first_to_new.inverse();
    new_frame.shell->aff_light = AffLight(0, 0);
    new_frame.SetZeroEvalPT(new_frame.shell->cam_to_world.inverse(),
                            new_frame.shell->aff_light);
    new_frame.shell->tracking_ref = first_frame->shell;
    new_frame.shell->cam_to_ref = first_to_new.inverse();
  }
  initialized = true;
  LOG(INFO) << "INITIALIZE FROM INITIALIZER ("
            << (int)first_frame->point_hessians.size() << " pts)";
  coarse_tracker_for_new_kf->MakeK(HCalib);
  coarse_tracker_for_new_kf->SetCoarseTrackingRef(frame_hessians);
}

void FullSystem::DeliverTrackedFrame(std::unique_ptr<FrameHessian> &fh,
                                     bool need_kf) {
  if (linearize_operation) {
    if (need_kf) {
      timing::Timer keyframe_timer("dsl/keyframe");
      MakeKeyFrame(fh);
      keyframe_timer.Stop();
    } else {
      timing::Timer non_keyframe_timer("dsl/non-keyframe");
      MakeNonKeyFrame(fh);
      non_keyframe_timer.Stop();
    }
  } else {
    // TODO: LINEARIZE_OPERATION
  }
}

void FullSystem::MakeKeyFrame(std::unique_ptr<FrameHessian> &fh) {
  FrameHessian *fh_ptr = fh.get();
  {
    // TODO: LINEARIZE_OPERATION
    assert(fh_ptr->shell->tracking_ref != nullptr);
    fh_ptr->shell->cam_to_world =
        fh_ptr->shell->tracking_ref->cam_to_world * fh_ptr->shell->cam_to_ref;
    fh_ptr->SetZeroEvalPT(fh_ptr->shell->cam_to_world.inverse(),
                          fh_ptr->shell->aff_light);
  }

  timing::Timer trace_timer("dsl/keyframe/tracing");
  TraceNewCoarse(*fh_ptr);
  trace_timer.Stop();
  // TODO: LINEARIZE_OPERATION

  FlagFramesForMarginalization();

  fh_ptr->idx = frame_hessians.size();
  frame_hessians.emplace_back(std::move(fh));
  fh_ptr->frame_id = all_kf_shells.size();
  all_kf_shells.push_back(fh_ptr->shell);
  ef->InsertFrame(fh_ptr, HCalib);

  timing::Timer setpre_timer("dsl/keyframe/setpre");
  SetPreCalcValues();
  setpre_timer.Stop();

  /// add new residuals for old points with new frame (fh)
  // NOTE: set only old points from the old frames, forward
  // NOTE: !!! new ph with homo can be added here to the new keyframe points
  int num_fwd_res_added = 0;
  for (std::unique_ptr<FrameHessian> &fh1 : frame_hessians) {
    if (fh1.get() == fh_ptr) continue;
    for (std::unique_ptr<PointHessian> &ph : fh1->point_hessians) {
      // NOTE: HomoResidual constraints can be added here from fh1 as host, to
      // fh as new frame
      std::unique_ptr<PointFrameResidual> r =
          std::make_unique<PointFrameResidual>(ph.get(), fh1.get(), fh_ptr,
                                               &HCalib);
      // std::make_unique<HomoResidual>(ph, fh1, fh, settingBaseline);
      PointFrameResidual *r_ptr = r.get();
      r_ptr->SetState(ResState::IN);
      ef->InsertResidual(r.get());
      r->point->residuals.emplace_back(std::move(r));
      ph->last_residuals[1] = ph->last_residuals[0];
      ph->last_residuals[0] =
          std::pair<PointFrameResidual *, ResState>(r_ptr, ResState::IN);
      num_fwd_res_added += 1;
    }
  }

  timing::Timer activate_timer("dsl/keyframe/activate");
  ActivatePointsMT();
  ef->MakeIdx();
  activate_timer.Stop();

  LOG(INFO) << "before last ab:"
            << frame_hessians[frame_hessians.size() - 2]->GetAffLight().a
            << ", "
            << frame_hessians[frame_hessians.size() - 2]->GetAffLight().b;
  LOG(INFO) << "before current ab:" << frame_hessians.back()->GetAffLight().a
            << ", " << frame_hessians.back()->GetAffLight().b;

  timing::Timer opt_timer("dsl/keyframe/optimization");
  last_rmse = Optimize(settingMaxOptIterations);
  opt_timer.Stop();

  LOG(INFO) << "after last ab:"
            << frame_hessians[frame_hessians.size() - 2]->GetAffLight().a
            << ", "
            << frame_hessians[frame_hessians.size() - 2]->GetAffLight().b;
  LOG(INFO) << "after current ab:" << frame_hessians.back()->GetAffLight().a
            << ", " << frame_hessians.back()->GetAffLight().b;

  if (all_kf_shells.size() <= 4) {
    if (all_kf_shells.size() == 2 &&
        last_rmse > 20 * benchmarkInitializerSlackFactor) {
      LOG(WARNING) << ("I THINK INITIALIZATINO FAILED! Resetting.\n");
      init_failed = true;
    }
    if (all_kf_shells.size() == 3 &&
        last_rmse > 13 * benchmarkInitializerSlackFactor) {
      LOG(WARNING) << ("I THINK INITIALIZATINO FAILED! Resetting.\n");
      init_failed = true;
    }
    if (all_kf_shells.size() == 4 &&
        last_rmse > 9 * benchmarkInitializerSlackFactor) {
      LOG(WARNING) << "rmse: " << last_rmse;
      LOG(WARNING) << ("I THINK INITIALIZATINO FAILED! Resetting.\n");
      init_failed = true;
    }
  }

  if (is_lost) return;

  // Remove outliers, which should not be involved in the marginalization
  RemoveOutliers();

  {
    // TODO: LINEARIZE_OPERATION
    coarse_tracker_for_new_kf->MakeK(HCalib);
    coarse_tracker_for_new_kf->SetCoarseTrackingRef(frame_hessians);
    /*
    {
      int lvl = 0;
      LOG(INFO) << hG[lvl] << " " << wG[lvl] << " " << pyrLevelsUsed;
      cv::Mat show_img = cv::Mat(hG[lvl], wG[lvl], CV_8UC3);
      for (int x = 0; x < wG[0]; ++x) {
        for (int y = 0; y < hG[0]; ++y) {
          float c =
              coarse_tracker_for_new_kf->last_ref_fh->dI[x + y * wG[0]].x();
          show_img.at<cv::Vec3b>(y, x) = cv::Vec3b(c, c, c);
        }
      }

      for (int x = 0; x < wG[0]; ++x) {
        for (int y = 0; y < hG[0]; ++y) {
          float idist = coarse_tracker_for_new_kf->idist[lvl][x + y * wG[0]];
          if (idist > 0) {
            Vec3b color_idist = MakeJet3B(idist);
            cv::drawMarker(
                show_img, cv::Point(x, y),
                cv::Scalar(color_idist[2], color_idist[1], color_idist[0]),
                cv::MARKER_CROSS, 5, 1);
          }
        }
      }

      cv::namedWindow("idist", cv::WINDOW_AUTOSIZE);
      cv::imshow("idist", show_img);

      cv::waitKey(1);
    }
     */
  }

  FlagPointsForRemoval();
  ef->DropPointsF();

  if (settigEnableMarginalization) {
    timing::Timer marg_timer("dsl/keyframe/marginalization");
    Marginalization();
    marg_timer.Stop();
  }

  for (std::unique_ptr<FrameHessian> &host : frame_hessians) {
    for (auto &&ph : host->point_hessians_marginalized) {
      PointHessian *ph_ptr = ph.get();
      if (ph_ptr == nullptr) continue;
      if (ph_ptr->ef_point &&
          ph_ptr->ef_point->state_flag == EfPointStatus::MARGINALIZE) {
        for (auto &&r : ph_ptr->residuals) {
          if (r->res_blk_spec->residual_block_id) {
            parameter_map->RemoveResidualBlockSpec(r->res_blk_spec.get());
          }
        }
        parameter_map->RemoveParameterBlockSpec(&ph_ptr->parameter_idist);
      }
    }
  }

  ef->MarginalizePointsF();

  timing::Timer newtraces_timer("dsl/keyframe/new_traces");
  MakeNewTraces(*fh_ptr);
  newtraces_timer.Stop();

  for (unsigned int i = 0; i < frame_hessians.size(); ++i) {
    if (frame_hessians[i]->flagged_for_marginalization) {
      MarginalizeFrame(frame_hessians[i]);
      --i;
    }
  }

  LOG(INFO) << timing::Timing::Print();
}
void FullSystem::MakeNonKeyFrame(std::unique_ptr<FrameHessian> &fh) {
  {
    std::unique_lock<std::mutex> lock(shell_pose_mutex);
    assert(fh->shell->tracking_ref != 0);
    fh->shell->cam_to_world =
        fh->shell->tracking_ref->cam_to_world * fh->shell->cam_to_ref;
    fh->SetZeroEvalPT(fh->shell->cam_to_world.inverse(), fh->shell->aff_light);
  }

  TraceNewCoarse(*fh);
}
void FullSystem::MappingLoop() {
  // TODO: LINEARIZE_OPERATION
}

void FullSystem::SetPreCalcValues() {
  for (std::unique_ptr<FrameHessian> &fh : frame_hessians) {
    fh->target_precalc.resize(frame_hessians.size());
    for (unsigned int i = 0; i < frame_hessians.size(); ++i) {
      fh->target_precalc[i].Set(fh.get(), frame_hessians[i].get(), HCalib);
    }
  }

  // WARNING: I don't think deltaF should exist in this framework
}

void FullSystem::SetNewFrameEnergyTh() {
  std::vector<float> all_res_vec;
  all_res_vec.reserve(active_residuals.size() * 2);
  FrameHessian *newFrame = frame_hessians.back().get();
  DLOG(INFO) << "newFrame->frameEnergyTH before: " << newFrame->frame_energy_th;

  for (PointFrameResidual *r : active_residuals)
    if (r->new_res_energy_with_outiler >= 0 && r->target == newFrame) {
      all_res_vec.push_back(r->new_res_energy_with_outiler);
    }

  if (all_res_vec.size() == 0) {
    newFrame->frame_energy_th = 12 * 12 * patternNum;
    return;  // should never happen, but lets make sure.
  }

  int nthIdx = settingFrameEnergyThN * all_res_vec.size();

  assert(nthIdx < (int)all_res_vec.size());
  assert(settingFrameEnergyThN < 1);

  std::nth_element(all_res_vec.begin(), all_res_vec.begin() + nthIdx,
                   all_res_vec.end());
  float nth_element = sqrtf(all_res_vec[nthIdx]);

  newFrame->frame_energy_th = nth_element * settingFrameEnergyThFacMedian;
  DLOG(INFO) << "nth_element * settingFrameEnergyThFacMedian: "
             << nth_element * settingFrameEnergyThFacMedian;
  newFrame->frame_energy_th =
      26.0f * settingFrameEnergyThConstWeight +
      newFrame->frame_energy_th * (1 - settingFrameEnergyThConstWeight);
  newFrame->frame_energy_th =
      newFrame->frame_energy_th * newFrame->frame_energy_th;
  newFrame->frame_energy_th *=
      settingOverallEnergyThWeight * settingOverallEnergyThWeight;
  LOG(INFO) << "newFrame->frameEnergyTH: " << newFrame->frame_energy_th;
}

}  // namespace dsl