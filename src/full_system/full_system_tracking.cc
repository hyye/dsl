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

Vec4 FullSystem::TrackNewCoarse(FrameHessian &fh) {
  assert(all_frame_shells.size() > 0);

  timing::Timer coarse_tracker_timer("dsl/add_frame/coarse_tracker");

  FrameHessian *last_frame = coarse_tracker->last_ref_fh;

  AffLight aff_last_to_l = AffLight(0, 0);

  // WARNING: c++17 is magic?
  std::vector<SE3> last_frame_to_fh_tries;  /// potential pose tries
  if (all_frame_shells.size() == 2) {
    // doing nothing here
    last_frame_to_fh_tries.push_back(SE3());
  } else {
    FrameShell *slast = all_frame_shells[all_frame_shells.size() - 2].get();
    FrameShell *sprelast = all_frame_shells[all_frame_shells.size() - 3].get();
    SE3 slast_to_sprelast;
    SE3 last_frame_to_slast;
    {
      // TODO: LINEARIZE_OPERATION
      slast_to_sprelast =
          sprelast->cam_to_world.inverse() * slast->cam_to_world;
      /// from Keyframe last_frame
      last_frame_to_slast =
          slast->cam_to_world.inverse() * last_frame->shell->cam_to_world;
      aff_last_to_l = slast->aff_light;
      // aff_last_to_l = last_frame->shell->aff_light;
    }
    // assumed to be the same as fh_to_slast.
    SE3 fh_to_slast = slast_to_sprelast;

    // get last delta-movement.
    // assume constant motion.
    last_frame_to_fh_tries.push_back(fh_to_slast.inverse() *
                                     last_frame_to_slast);
    // assume double motion (frame skipped)
    last_frame_to_fh_tries.push_back(
        fh_to_slast.inverse() * fh_to_slast.inverse() * last_frame_to_slast);
    // assume half motion.
    last_frame_to_fh_tries.push_back(
        SE3::exp(fh_to_slast.log() * 0.5).inverse() * last_frame_to_slast);
    // assume zero motion.
    last_frame_to_fh_tries.push_back(last_frame_to_slast);
    // assume zero motion FROM KF.
    last_frame_to_fh_tries.push_back(SE3());

    // just try a TON of different initializations (all rotations). In the end,
    // if they don't work they will only be tried on the coarsest level, which
    // is super fast anyway. also, if tracking rails here we loose, so we
    // really, really want to avoid that. total 3*2 + 3*4 + 1*8 = 26 rotations
    for (float rotDelta = 0.02; rotDelta < 0.05; ++rotDelta) {
      last_frame_to_fh_tries.push_back(
          fh_to_slast.inverse() * last_frame_to_slast *
          SE3(Eigen::Quaterniond(1, rotDelta, 0, 0),
              Vec3(0, 0,
                   0)));  // assume constant motion.
      last_frame_to_fh_tries.push_back(
          fh_to_slast.inverse() * last_frame_to_slast *
          SE3(Eigen::Quaterniond(1, 0, rotDelta, 0),
              Vec3(0, 0,
                   0)));  // assume constant motion.
      last_frame_to_fh_tries.push_back(
          fh_to_slast.inverse() * last_frame_to_slast *
          SE3(Eigen::Quaterniond(1, 0, 0, rotDelta),
              Vec3(0, 0,
                   0)));  // assume constant motion.
      last_frame_to_fh_tries.push_back(
          fh_to_slast.inverse() * last_frame_to_slast *
          SE3(Eigen::Quaterniond(1, -rotDelta, 0, 0),
              Vec3(0, 0,
                   0)));  // assume constant motion.
      last_frame_to_fh_tries.push_back(
          fh_to_slast.inverse() * last_frame_to_slast *
          SE3(Eigen::Quaterniond(1, 0, -rotDelta, 0),
              Vec3(0, 0,
                   0)));  // assume constant motion.
      last_frame_to_fh_tries.push_back(
          fh_to_slast.inverse() * last_frame_to_slast *
          SE3(Eigen::Quaterniond(1, 0, 0, -rotDelta),
              Vec3(0, 0,
                   0)));  // assume constant motion.
      last_frame_to_fh_tries.push_back(
          fh_to_slast.inverse() * last_frame_to_slast *
          SE3(Eigen::Quaterniond(1, rotDelta, rotDelta, 0),
              Vec3(0, 0, 0)));  // assume constant motion.
      last_frame_to_fh_tries.push_back(
          fh_to_slast.inverse() * last_frame_to_slast *
          SE3(Eigen::Quaterniond(1, 0, rotDelta, rotDelta),
              Vec3(0, 0, 0)));  // assume constant motion.
      last_frame_to_fh_tries.push_back(
          fh_to_slast.inverse() * last_frame_to_slast *
          SE3(Eigen::Quaterniond(1, rotDelta, 0, rotDelta),
              Vec3(0, 0, 0)));  // assume constant motion.
      last_frame_to_fh_tries.push_back(
          fh_to_slast.inverse() * last_frame_to_slast *
          SE3(Eigen::Quaterniond(1, -rotDelta, rotDelta, 0),
              Vec3(0, 0, 0)));  // assume constant motion.
      last_frame_to_fh_tries.push_back(
          fh_to_slast.inverse() * last_frame_to_slast *
          SE3(Eigen::Quaterniond(1, 0, -rotDelta, rotDelta),
              Vec3(0, 0, 0)));  // assume constant motion.
      last_frame_to_fh_tries.push_back(
          fh_to_slast.inverse() * last_frame_to_slast *
          SE3(Eigen::Quaterniond(1, -rotDelta, 0, rotDelta),
              Vec3(0, 0, 0)));  // assume constant motion.
      last_frame_to_fh_tries.push_back(
          fh_to_slast.inverse() * last_frame_to_slast *
          SE3(Eigen::Quaterniond(1, rotDelta, -rotDelta, 0),
              Vec3(0, 0, 0)));  // assume constant motion.
      last_frame_to_fh_tries.push_back(
          fh_to_slast.inverse() * last_frame_to_slast *
          SE3(Eigen::Quaterniond(1, 0, rotDelta, -rotDelta),
              Vec3(0, 0, 0)));  // assume constant motion.
      last_frame_to_fh_tries.push_back(
          fh_to_slast.inverse() * last_frame_to_slast *
          SE3(Eigen::Quaterniond(1, rotDelta, 0, -rotDelta),
              Vec3(0, 0, 0)));  // assume constant motion.
      last_frame_to_fh_tries.push_back(
          fh_to_slast.inverse() * last_frame_to_slast *
          SE3(Eigen::Quaterniond(1, -rotDelta, -rotDelta, 0),
              Vec3(0, 0, 0)));  // assume constant motion.
      last_frame_to_fh_tries.push_back(
          fh_to_slast.inverse() * last_frame_to_slast *
          SE3(Eigen::Quaterniond(1, 0, -rotDelta, -rotDelta),
              Vec3(0, 0, 0)));  // assume constant motion.
      last_frame_to_fh_tries.push_back(
          fh_to_slast.inverse() * last_frame_to_slast *
          SE3(Eigen::Quaterniond(1, -rotDelta, 0, -rotDelta),
              Vec3(0, 0, 0)));  // assume constant motion.
      last_frame_to_fh_tries.push_back(
          fh_to_slast.inverse() * last_frame_to_slast *
          SE3(Eigen::Quaterniond(1, -rotDelta, -rotDelta, -rotDelta),
              Vec3(0, 0, 0)));  // assume constant motion.
      last_frame_to_fh_tries.push_back(
          fh_to_slast.inverse() * last_frame_to_slast *
          SE3(Eigen::Quaterniond(1, -rotDelta, -rotDelta, rotDelta),
              Vec3(0, 0, 0)));  // assume constant motion.
      last_frame_to_fh_tries.push_back(
          fh_to_slast.inverse() * last_frame_to_slast *
          SE3(Eigen::Quaterniond(1, -rotDelta, rotDelta, -rotDelta),
              Vec3(0, 0, 0)));  // assume constant motion.
      last_frame_to_fh_tries.push_back(
          fh_to_slast.inverse() * last_frame_to_slast *
          SE3(Eigen::Quaterniond(1, -rotDelta, rotDelta, rotDelta),
              Vec3(0, 0, 0)));  // assume constant motion.
      last_frame_to_fh_tries.push_back(
          fh_to_slast.inverse() * last_frame_to_slast *
          SE3(Eigen::Quaterniond(1, rotDelta, -rotDelta, -rotDelta),
              Vec3(0, 0, 0)));  // assume constant motion.
      last_frame_to_fh_tries.push_back(
          fh_to_slast.inverse() * last_frame_to_slast *
          SE3(Eigen::Quaterniond(1, rotDelta, -rotDelta, rotDelta),
              Vec3(0, 0, 0)));  // assume constant motion.
      last_frame_to_fh_tries.push_back(
          fh_to_slast.inverse() * last_frame_to_slast *
          SE3(Eigen::Quaterniond(1, rotDelta, rotDelta, -rotDelta),
              Vec3(0, 0, 0)));  // assume constant motion.
      last_frame_to_fh_tries.push_back(
          fh_to_slast.inverse() * last_frame_to_slast *
          SE3(Eigen::Quaterniond(1, rotDelta, rotDelta, rotDelta),
              Vec3(0, 0, 0)));  // assume constant motion.
    }

    if (!slast->pose_valid || !sprelast->pose_valid ||
        !last_frame->shell->pose_valid) {
      last_frame_to_fh_tries.clear();
      last_frame_to_fh_tries.push_back(SE3());
    }
  }

  Vec3 flow_vecs = Vec3(100, 100, 100);
  SE3 last_frame_to_fh = SE3();
  AffLight aff_light = AffLight(0, 0);

  SE3 best_last_frame_to_fh = SE3();
  AffLight best_aff_light = AffLight(0, 0);

  // as long as maxResForImmediateAccept is not reached, I'll continue through
  // the options. I'll keep track of the so-far best achieved residual for each
  // level in achieved_res. If on a coarse level, tracking is WORSE than
  // achieved_res, we will not continue to save time.
  Vec5 achieved_res = Vec5::Constant(NAN);
  Vec5 best_res = Vec5::Constant(NAN);
  bool have_one_good = false;
  int try_iterations = 0;

  DLOG(INFO) << "aff_last: " << aff_last_to_l.a << ", " << aff_last_to_l.b;

  for (unsigned int i = 0; i < last_frame_to_fh_tries.size(); i++) {
    AffLight aff_light_this = aff_last_to_l;
    SE3 last_frame_to_fh_this = last_frame_to_fh_tries[i];

    DLOG(INFO) << "last_frame_to_fh_this: " << std::endl
               << last_frame_to_fh_this.matrix3x4();
    DLOG(INFO) << "aff_last_to_l: " << aff_last_to_l.a << " "
               << aff_last_to_l.b;

    // track using this try
    // in each level has to be at least as good as the last try.
    timing::Timer try_timer("dsl/add_frame/coarse_tracker/try");
    bool tracking_is_good = coarse_tracker->TrackNewestCoarse(
        fh, last_frame_to_fh_this, aff_light_this, pyrLevelsUsed - 1,
        achieved_res);
    try_timer.Stop();
    try_iterations++;

    if (i != 0) {
      char buff[1000];
      snprintf(
          buff, sizeof(buff),
          "RE-TRACK ATTEMPT %d with initOption %d and start-lvl %d (ab %f %f): "
          "%f %f %f %f %f -> %f %f %f %f %f \n",
          i, i, pyrLevelsUsed - 1, aff_light_this.a, aff_light_this.b,
          achieved_res[0], achieved_res[1], achieved_res[2], achieved_res[3],
          achieved_res[4], coarse_tracker->last_residuals[0],
          coarse_tracker->last_residuals[1], coarse_tracker->last_residuals[2],
          coarse_tracker->last_residuals[3], coarse_tracker->last_residuals[4]);
      std::string buff_as_str = buff;
      LOG(INFO) << buff_as_str;
    }

    // do we have a new winner?
    if (tracking_is_good &&
        std::isfinite((float)coarse_tracker->last_residuals[0]) &&
        !(coarse_tracker->last_residuals[0] >= achieved_res[0])) {
      DLOG(WARNING) << "good! "
                    << "(float)coarse_tracker->last_residuals[0]: "
                    << (float)coarse_tracker->last_residuals[0]
                    << " achieved_res[0]: " << achieved_res[0]
                    << " last_coarse_rmse[0]: " << last_coarse_rmse[0];

      flow_vecs = coarse_tracker->last_flow_indicators;
      aff_light = aff_light_this;
      last_frame_to_fh = last_frame_to_fh_this;
      have_one_good = true;
    } else {
      DLOG(WARNING) << "not good! "
                   << "(float)coarse_tracker->last_residuals[0]:"
                   << (float)coarse_tracker->last_residuals[0]
                   << " achieved_res[0]:" << achieved_res[0];
    }

    // take over achieved res (always).
    if (have_one_good && tracking_is_good) {
      for (int res_i = 0; res_i < 5; res_i++) {
        // take over if achieved_res is either bigger or NAN.
        if (!std::isfinite((float)achieved_res[res_i]) ||
            achieved_res[res_i] > coarse_tracker->last_residuals[res_i])
          achieved_res[res_i] = coarse_tracker->last_residuals[res_i];
      }
    }

    if (have_one_good &&
        achieved_res[0] < last_coarse_rmse[0] * settingReTrackThreshold) {
      timing::Timer t("tracker/t");
      DLOG(INFO) << achieved_res[0] << " @T@ " << last_coarse_rmse[0] * settingReTrackThreshold;
      break;
    } else {
      timing::Timer t("tracker/f");
      DLOG(INFO) << achieved_res[0] << " @F@ " << last_coarse_rmse[0] * settingReTrackThreshold;
    }
  }

  if (!have_one_good) {
    LOG(ERROR) << "BIG ERROR! tracking failed entirely. Take predicted pose "
                  "and hope we may somehow recover.";
    flow_vecs = Vec3(0, 0, 0);
    aff_light = aff_last_to_l;
    last_frame_to_fh = last_frame_to_fh_tries[0];
  }

  // LOG(INFO) << "tracked aff_light: " << aff_light.a << ", " << aff_light.b << " try_iterations: " << try_iterations;

  last_coarse_rmse = achieved_res;

  // no lock required, as fh is not used anywhere yet.
  fh.shell->cam_to_ref = last_frame_to_fh.inverse();
  fh.shell->tracking_ref = last_frame->shell;
  fh.shell->aff_light = aff_light;
  fh.shell->cam_to_world =
      fh.shell->tracking_ref->cam_to_world * fh.shell->cam_to_ref;

  if (coarse_tracker->first_coarse_rmse < 0) {
    coarse_tracker->first_coarse_rmse = achieved_res[0];
  }

  if (!settingDebugoutRunquiet) {
    char buff[1000];
    snprintf(buff, sizeof(buff),
             "Coarse Tracker tracked ab = %f %f (exp %f). Res %f!\n",
             aff_light.a, aff_light.b, fh.exposure, achieved_res[0]);
    std::string buff_as_str = buff;
    LOG(INFO) << buff_as_str;
  }

  DLOG(INFO) << "try_iterations: " << try_iterations;

  return Vec4(achieved_res[0], flow_vecs[0], flow_vecs[1], flow_vecs[2]);
}

}  // namespace dsl