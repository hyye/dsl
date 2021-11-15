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

#include "ceres/ceres.h"
#include "full_system/full_system.h"
#include "optimization/homo_cost_functor.h"
#include "optimization/se3_local_parameterization.h"
#include "optimization/se3_prior_cost_functor.h"
#include "tool/cv_helper.h"
#include "full_system/optimization_call_back.h"

namespace dsl {

void FullSystem::StateToParameter() {
  // for (int i = 0; i < frame_hessians.size(); ++i) {
  //   parameter_poses_[i] = frame_hessians[i]->PRE_world_to_cam.log();
  //   parameter_abs_[i] = frame_hessians[i]->GetAffLight().Vec();
  // }
}

void FullSystem::ParameterToState() {
  // WARNING: no exposure time considered
  // SE3 pose00 = frame_hessians[0]->PRE_world_to_cam;
  // Vec2 ab00 = frame_hessians[0]->GetAffLight().Vec();
  // SE3 pose0 = SE3::exp(parameter_poses_[0]);
  // Vec2 ab0 = parameter_abs_[0];
  // for (int i = 0; i < frame_hessians.size(); ++i) {
  //   parameter_poses_[i] =
  //       (SE3::exp(parameter_poses_[i]) * pose0.inverse() * pose00).log();
  //   Vec2 abi = parameter_abs_[i];
  //   parameter_abs_[i].y() =
  //       abi.y() - exp(abi.x() - ab0.x()) * (ab0.y() - ab00.y());
  //   parameter_abs_[i].x() = abi.x() - ab0.x() + ab00.x();
  // }
}

float FullSystem::Optimize(int max_opt_its) {
  if (frame_hessians.size() < 2) return 0;
  if (frame_hessians.size() < 3) max_opt_its = 20;
  if (frame_hessians.size() < 4) max_opt_its = 15;

  timing::Timer pre_pre_timer("dsl/keyframe/optimization/pre_pre");
  active_residuals.clear();
  int num_points = 0;
  int num_lin_res = 0;

  for (std::unique_ptr<FrameHessian> &fh : frame_hessians) {
    for (std::unique_ptr<PointHessian> &ph : fh->point_hessians) {
      for (std::unique_ptr<PointFrameResidual> &pfr : ph->residuals) {
        if (!pfr->ef_residual->is_linearized) {
          active_residuals.push_back(pfr.get());
          pfr->ResetOOB();
        } else {
          ++num_lin_res;
        }
      }
      ++num_points;
    }
  }

  LOG(INFO) << "num_points: " << num_points;

  int num_ef_points = 0;
  for (auto &&ef_frame : ef->ef_frames) {
    num_ef_points += ef_frame->ef_points.size();
  }
  DLOG(INFO) << "num_ef_points: " << num_ef_points;

  if (!settingDebugoutRunquiet) {
    char buff[100];
    snprintf(buff, sizeof(buff),
             "OPTIMIZE %d pts, %d active res, %d lin res!\n", ef->num_points,
             (int)active_residuals.size(), num_lin_res);
    std::string buff_as_str = buff;
    LOG(INFO) << buff_as_str;
  }
  pre_pre_timer.Stop();

  double final_cost;

  {
    timing::Timer pre_timer("dsl/keyframe/optimization/pre");
    std::vector<double *> param_ids;
    std::vector<ceres::ResidualBlockId> res_ids_marg;
    std::vector<ceres::ResidualBlockId> res_ids_photo;
    for (int i = 0; i < frame_hessians.size(); ++i) {
      frame_hessians[i]->parameter_pose.SetEstimate(
          frame_hessians[i]->PRE_world_to_cam.log());
      frame_hessians[i]->parameter_ab.SetEstimate(
          frame_hessians[i]->GetAffLight().Vec());
      parameter_map->AddParameterBlockSpec(&frame_hessians[i]->parameter_pose,
                                           se3_local_parameterization.get(), 1);
      parameter_map->AddParameterBlockSpec(&frame_hessians[i]->parameter_ab,
                                           ab_local_parameterization.get(), 1);
      param_ids.push_back(frame_hessians[i]->parameter_pose.GetParameters());
      param_ids.push_back(frame_hessians[i]->parameter_ab.GetParameters());
    }
    // WARNING: fix the first frame
    if (!no_prior_ || (frame_hessians.size() < 3)) {
      parameter_map->problem->SetParameterBlockConstant(
          frame_hessians[0]->parameter_pose.parameters);
      // parameter_map->problem->SetParameterBlockConstant(
      //     frame_hessians[0]->parameter_ab.parameters);
    }

    // if (frame_hessians.size() == 2) {
    //   Eigen::Matrix<double, 6, 6> information =
    //       Eigen::Matrix<double, 6, 6>::Identity() * 1e4;
    //   frame_hessians.front()->se3_prior_cost_function =
    //       std::make_unique<SE3PriorCostFunctor>(
    //           frame_hessians.front()->PRE_world_to_cam, information);
    //   ParameterMap::ParameterBlockCollection parameter_block_collection{
    //       &frame_hessians.front()->parameter_pose};
    //   parameter_map->AddResidualBlockSpec(
    //       &frame_hessians.front()->res_blk_spec,
    //       frame_hessians.front()->se3_prior_cost_function.get(), NULL,
    //       parameter_block_collection);
    // }

    /// Add marginalization residual
    // FIXME: uncomment
    if (marginalization_info_->valid) {
      // FIXME: duplicate
      if (marginalization_res_ &&
          marginalization_res_->res_blk_spec->residual_block_id) {
        parameter_map->RemoveResidualBlockSpec(
            marginalization_res_->res_blk_spec.get());
      }

      marginalization_res_ = std::make_unique<MarginalizationResidual>(
          marginalization_info_.get());
      parameter_map->AddResidualBlockSpec(
          marginalization_res_->res_blk_spec.get(),
          marginalization_res_->cost_function.get(), NULL,
          marginalization_info_->keep_block_collection);
      res_ids_marg.push_back(
          marginalization_res_->res_blk_spec->residual_block_id);
    }

    DLOG(INFO) << "ef->num_points: " << ef->num_points;

    int idx_in_param = 0;
    for (int i = 0; i < active_residuals.size(); ++i) {
      PointFrameResidual *res = active_residuals[i];
      active_residuals[i]->point->parameter_idist.SetEstimate(
          active_residuals[i]->point->idist);
      parameter_map->AddParameterBlockSpec(
          &active_residuals[i]->point->parameter_idist, nullptr, 0);
      param_ids.push_back(
          active_residuals[i]->point->parameter_idist.GetParameters());

      if (!res->res_blk_spec->residual_block_id) {
        ParameterMap::ParameterBlockCollection parameter_block_collection{
            &res->target->parameter_pose, &res->target->parameter_ab,
            &res->host->parameter_pose, &res->host->parameter_ab,
            &res->point->parameter_idist};
        parameter_map->AddResidualBlockSpec(res->res_blk_spec.get(),
                                            res->cost_function.get(), NULL,
                                            parameter_block_collection);
      }
      res_ids_photo.push_back(res->res_blk_spec->residual_block_id);
      if (res->idist_converged) {
        parameter_map->problem->SetParameterBlockConstant(
            res->point->parameter_idist.GetParameters());
      }
    }

    auto PrintAb = [](std::vector<std::unique_ptr<FrameHessian>> &fhs) {
      std::stringstream ss;
      ss << "ab: ";
      for (int i = 0; i < fhs.size(); ++i) {
        ss << std::fixed << std::setprecision(3)
           << fhs[i]->parameter_ab.GetEstimate().transpose() << "; ";
      }
      ss << std::endl;
      LOG(INFO) << ss.str();
    };

    DLOG(INFO) << "idx_in_param: " << idx_in_param;
    LOG(INFO) << "before:" << std::endl
              << SE3::exp(frame_hessians.back()->parameter_pose.GetEstimate())
                     .inverse()
                     .matrix3x4();
    PrintAb(frame_hessians);
    pre_timer.Stop();

    // {
    //   double cost_marg = 0.0, cost_photo = 0.0;
    //   ceres::Problem::EvaluateOptions e_option;
    //   e_option.parameter_blocks = param_ids;
    //   e_option.residual_blocks = res_ids_photo;
    //   parameter_map->problem->Evaluate(e_option, &cost_photo, NULL, NULL, NULL);
    //   if (!res_ids_marg.empty()) {
    //     e_option.residual_blocks = res_ids_marg;
    //     parameter_map->problem->Evaluate(e_option, &cost_marg, NULL, NULL,
    //                                      NULL);
    //   }
    //   LOG(INFO) << ">>>>>>> BEFORE <<<<<<< cost_photo: " << cost_photo
    //             << " cost_marg: " << cost_marg;
    // }

    parameter_map->options.num_threads = NUM_THREADS;
    parameter_map->options.linear_solver_type =
        ceres::DENSE_SCHUR;  // since # of cam is small
    parameter_map->options.max_num_iterations = max_opt_its;
    parameter_map->options.update_state_every_iteration = true;
    parameter_map->options.trust_region_strategy_type = ceres::DOGLEG;
    parameter_map->options.dogleg_type = ceres::SUBSPACE_DOGLEG;
    parameter_map->options.callbacks.clear();
    std::unique_ptr<ceres::IterationCallback> cb =
        std::make_unique<OptimizationCallBack>(*this);
    parameter_map->options.callbacks.push_back(cb.get());
    // options.minimizer_progress_to_stdout = true;

    timing::Timer solve_timer("dsl/keyframe/optimization/solve");
    parameter_map->SolveProblem();
    solve_timer.Stop();

    {
      double cost_marg = 0.0, cost_photo = 0.0;
      ceres::Problem::EvaluateOptions e_option;
      e_option.parameter_blocks = param_ids;
      e_option.residual_blocks = res_ids_photo;
      parameter_map->problem->Evaluate(e_option, &cost_photo, NULL, NULL, NULL);
      // if (!res_ids_marg.empty()) {
      //   e_option.residual_blocks = res_ids_marg;
      //   parameter_map->problem->Evaluate(e_option, &cost_marg, NULL, NULL,
      //                                    NULL);
      // }
      // LOG(INFO) << ">>>>>>> AFTER <<<<<<< cost_photo: " << cost_photo
      //           << " cost_marg: " << cost_marg;
      final_cost = cost_photo;
    }

    DLOG(INFO) << parameter_map->summary.FullReport() << "\n";

    LOG(INFO) << "after:" << std::endl
              << SE3::exp(frame_hessians.back()->parameter_pose.GetEstimate())
                     .inverse()
                     .matrix3x4();
    PrintAb(frame_hessians);

    // final_cost = parameter_map->summary.final_cost;
    LOG(INFO) << ">./>./>./frame_hessians.size(): " << frame_hessians.size();

    {
      LOG(INFO) << "<<<<<<<<<<<<<<";
      int count_in = -1, count_all = 0;
      for (auto &&r : active_residuals) {
        ++count_all;
        if (r->res_state == ResState::IN) {
          ++count_in;
        }
      }
      LOG(INFO) << float(count_in) / count_all << " " << count_in << "/"
                << count_all;
    }

    /// Set updated states here
    for (int i = 0; i < frame_hessians.size(); ++i) {
      frame_hessians[i]->SetState(
          frame_hessians[i]->parameter_pose.GetEstimate(),
          frame_hessians[i]->parameter_ab.GetEstimate());
    }
    for (int i = 0; i < active_residuals.size(); ++i) {
      if (active_residuals[i]->idist_converged) {
        active_residuals[i]->point->idist = active_residuals[i]->ph_idist;
      } else {
        active_residuals[i]->point->idist =
            active_residuals[i]->point->parameter_idist.GetEstimate();
      }
    }

    // std::stringstream ss_hessian;
    // ss_hessian << "Hdd: ";
    for (std::unique_ptr<FrameHessian> &fh : frame_hessians) {
      for (std::unique_ptr<PointHessian> &ph : fh->point_hessians) {
        double Hdd = 0;
        int num_good_res = 0;
        for (std::unique_ptr<PointFrameResidual> &pfr : ph->residuals) {
          if (!pfr->ef_residual->is_linearized &&
              pfr->res_state == ResState::IN) {
            Hdd += pfr->Jd.transpose() * pfr->Jd;
            ++num_good_res;
          }
        }
        if (num_good_res > 0) {
          if (Hdd < 1e-10) Hdd = 1e-10;
          ph->ef_point->HdiF = 1.0 / Hdd;
          ph->idist_hessian = Hdd;
        } else {
          ph->ef_point->HdiF = ph->idist_hessian = 0;
        }
        // ss_hessian << Hdd << ", ";
      }
    }
    // LOG(INFO) << ss_hessian.str();

    /// Set state zero
    Vec10 new_state_zero = Vec10::Zero();
    new_state_zero.segment<2>(6) =
        frame_hessians.back()->GetState().segment<2>(6);
    frame_hessians.back()->SetEvalPT(frame_hessians.back()->PRE_world_to_cam,
                                     new_state_zero);
    frame_hessians.back()->parameter_pose.SetLinearizationPoint();
    frame_hessians.back()->parameter_ab.SetLinearizationPoint();
  }

  timing::Timer post_timer("dsl/keyframe/optimization/post");
  int res_in_a = 0;
  int res_total = 0;
  std::vector<PointFrameResidual *> to_remove;
  for (PointFrameResidual *r : active_residuals) {
    PointHessian *ph = r->point;
    if (ph->last_residuals[0].first == r)
      ph->last_residuals[0].second = r->res_state;
    else if (ph->last_residuals[1].first == r)
      ph->last_residuals[1].second = r->res_state;

    EfResidual *efr = r->ef_residual;
    ++res_total;

    if (!efr->IsActive()) {
      to_remove.emplace_back(r);
      continue;
    } else {
      // WARNING: linearizeALL_Reductor
      ++r->point->num_good_residuals;
    }
    ++res_in_a;
  }

  int num_res_removed = 0;
  for (PointFrameResidual *r : to_remove) {
    if (r->res_blk_spec->residual_block_id) {
      parameter_map->RemoveResidualBlockSpec(r->res_blk_spec.get());
    }

    PointHessian *ph = r->point;

    if (ph->last_residuals[0].first == r)
      ph->last_residuals[0].first = nullptr;
    else if (ph->last_residuals[1].first == r)
      ph->last_residuals[1].first = nullptr;

    for (unsigned int k = 0; k < ph->residuals.size(); k++)
      if (ph->residuals[k].get() == r) {
        ef->DropResidual(r->ef_residual);
        DeleteOut<PointFrameResidual>(ph->residuals, k);
        num_res_removed++;
        break;
      }
  }
  LOG(INFO) << "toRemove: " << to_remove.size()
            << " num_res_removed: " << num_res_removed;

  LOG(INFO) << "res_in_a: " << res_in_a << " res_total: " << res_total;
  float rmse = sqrt(final_cost / (patternNum * res_in_a));
  LOG(INFO) << "rmse: " << rmse;
  LOG_IF(ERROR, rmse < 20) << "RMSE too large";
  // sqrtf((float)(lastEnergy[0] / (patternNum * ef->resInA)));

  // update frame_hessian after optimization
  {
    // TODO: LINEARIZE_OPERATION
    for (std::unique_ptr<FrameHessian> &fh : frame_hessians) {
      fh->shell->cam_to_world = fh->PRE_cam_to_world;
      fh->shell->aff_light = fh->GetAffLight();
    }
  }

  /*
  {
    std::vector<cv::Point2f> corners, corners_curr;
    for (PointFrameResidual *r : active_residuals) {
      PointHessian *ph = r->point;

      for (std::unique_ptr<EfResidual> &efr : ph->ef_point->ef_residuals) {
        if (!efr->IsActive()) continue;
        if (!(efr->target->frame == frame_hessians.back().get() &&
              efr->host->frame == frame_hessians.front().get()))
          continue;
        corners.emplace_back(ph->u, ph->v);
        corners_curr.emplace_back(efr->pfr->center_projected_to.x(),
                                  efr->pfr->center_projected_to.y());
      }
    }
    cv::Mat color_ref(hG[0], wG[0], CV_8UC3);
    cv::Mat color_curr(hG[0], wG[0], CV_8UC3);
    for (int x = 0; x < wG[0]; ++x) {
      for (int y = 0; y < hG[0]; ++y) {
        int i = x + y * wG[0];
        color_ref.at<cv::Vec3b>(y, x) =
            cv::Vec3b(frame_hessians.front()->dI[i].x(),
                      frame_hessians.front()->dI[i].x(),
                      frame_hessians.front()->dI[i].x());
        color_curr.at<cv::Vec3b>(y, x) = cv::Vec3b(
            frame_hessians.back()->dI[i].x(), frame_hessians.back()->dI[i].x(),
            frame_hessians.back()->dI[i].x());
      }
    }
    cv_helper::VisualizePairs(color_ref, color_curr, corners, corners_curr,
                              true, 100);
    cv::waitKey(0);
  }
  */

  return rmse;
}

void FullSystem::RemoveOutliers() {
  int num_points_dropped = 0;
  for (std::unique_ptr<FrameHessian> &fh : frame_hessians) {
    for (unsigned int i = 0; i < fh->point_hessians.size(); i++) {
      std::unique_ptr<PointHessian> &ph = fh->point_hessians[i];
      PointHessian *ph_ptr = ph.get();
      if (ph_ptr == nullptr) continue;

      if (ph_ptr->residuals.size() == 0) {
        fh->point_hessians_out.emplace_back(std::move(ph));
        ph_ptr->ef_point->state_flag = EfPointStatus::DROP;
        parameter_map->RemoveParameterBlockSpec(&ph_ptr->parameter_idist);
        std::swap(fh->point_hessians[i], fh->point_hessians.back());
        fh->point_hessians.pop_back();
        --i;
        ++num_points_dropped;
      }
    }
  }
  LOG(INFO) << "+++++++ num_points_dropped: " << num_points_dropped;
  ef->DropPointsF();
}

}  // namespace dsl
