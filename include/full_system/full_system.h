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

#ifndef DSL_FULLSYSTEM_H_
#define DSL_FULLSYSTEM_H_

#include "coarse_distance_map.h"
#include "coarse_tracker.h"
#include "distance_initializer.h"
#include "dsl_common.h"
#include "ef_struct.h"
#include "hessian_blocks.h"
#include "immature_point.h"
#include "optimization/marginalization_factor.h"
#include "pixel_selector.h"
#include "residual.h"
#include "marginalization_residual.h"
#include "util/index_thread_reduce.h"

namespace dsl {

class FullSystem {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  FullSystem();
  virtual ~FullSystem() = default;

  virtual void AddActiveFrame(
      ImageAndExposure &image, int id,
      const std::vector<float> &dist_metric = std::vector<float>(),
      const SE3 &pose_in_world = SE3(), Vec4f *vertex_map = nullptr,
      Vec4f *normal_map = nullptr);
  virtual void MarginalizeFrame(std::unique_ptr<FrameHessian> &frame);
  virtual float Optimize(int max_opt_its);

  virtual void RemoveOutliers();

  void SetNewFrameEnergyTh();

  virtual std::unique_ptr<PointHessian> OptimizeImmaturePoint(
      ImmaturePoint &point, int min_obs,
      std::vector<ImmaturePointTemporaryResidual> &residuals);

  void InitializeFromInitializer(FrameHessian &new_frame,
                                 Vec4f *vertex_map = nullptr,
                                 Vec4f *normal_map = nullptr);

  virtual Vec4 TrackNewCoarse(FrameHessian &fh);
  void TraceNewCoarse(FrameHessian &fh);
  void ActivatePointsMT();
  void ActivatePointsMTReductor(
      std::vector<std::unique_ptr<PointHessian>> *optimized,
      std::vector<std::unique_ptr<ImmaturePoint>> *to_optimize, int min,
      int max, Vec10 &stats, int tid);
  void MakeNewTraces(FrameHessian &new_frame);

  std::vector<FrameShell *> keyframe_shells;
  std::vector<std::unique_ptr<FrameHessian>> frame_hessians;
  std::vector<PointFrameResidual *> active_residuals;
  float current_min_act_dist;

  std::vector<float> all_res_vec;

  std::unique_ptr<EnergyFunction> ef;
  IndexThreadReduce<Vec10> thread_reduce;

  bool is_lost;
  bool init_failed;
  bool initialized;
  bool linearize_operation;

  std::mutex track_mutux;
  std::vector<std::unique_ptr<FrameShell>> all_frame_shells;
  std::unique_ptr<CoarseTracker> coarse_tracker;

  std::mutex coarse_tracker_swap_mutex;
  std::unique_ptr<CoarseTracker> coarse_tracker_for_new_kf;

  std::mutex map_mutex;
  std::vector<FrameShell *> all_kf_shells;

  std::unique_ptr<DistanceInitializer> distance_initializer;
  Vec5 last_coarse_rmse;

  std::vector<float> selection_map;
  std::unique_ptr<PixelSelector> pixel_selector;
  std::unique_ptr<CoarseDistanceMap> coarse_distance_map;

  std::mutex shell_pose_mutex;

  void DeliverTrackedFrame(std::unique_ptr<FrameHessian> &fh, bool need_kf);
  void MakeKeyFrame(std::unique_ptr<FrameHessian> &fh);
  void MakeNonKeyFrame(std::unique_ptr<FrameHessian> &fh);
  void MappingLoop();

  void FlagFramesForMarginalization();

  void FlagPointsForRemoval();

  void Marginalization();

  std::mutex track_map_sync_mutex;
  std::condition_variable track_frame_signal;
  std::condition_variable mapped_frame_signal;
  std::deque<std::unique_ptr<FrameHessian>> unmapped_tracked_frame;

  // Otherwise, a new KF is *needed that has ID bigger than[need_new_kf_after]*.
  int need_new_kf_after;
  std::thread mapping_thread;
  bool run_mapping;
  bool need_to_catchup_mapping;

  int last_ref_stop_id;

  bool is_keyframe = false;

  CalibHessian HCalib;

  void SetPreCalcValues();
  void ParameterToState();
  void StateToParameter();

  void SetNoPosePrior(bool no_prior) {
    no_prior_ = no_prior;
  }

  std::unique_ptr<ParameterMap> parameter_map;
  std::unique_ptr<ceres::LocalParameterization> se3_local_parameterization;
  std::unique_ptr<ceres::LocalParameterization> ab_local_parameterization;

  double last_rmse = 0;

 private:

  std::unique_ptr<MarginalizationInfo> marginalization_info_;
  std::unique_ptr<MarginalizationResidual> marginalization_res_;
  bool no_prior_ = true;
};

}  // namespace dsl

#endif  // DSL_FULLSYSTEM_H_
