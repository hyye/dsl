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
// Created by hyye on 11/13/19.
//

#include "full_system/full_system.h"
#include "optimization/homo_cost_functor.h"
#include "util/timing.h"

namespace dsl {

void FullSystem::FlagFramesForMarginalization() {
  /// if minFrameAge is set to be greater than maxFrames
  if (settingMinFrameAge > settingMaxFrames) {
    for (int i = settingMaxFrames; i < (int)frame_hessians.size(); ++i) {
      FrameHessian *fh = frame_hessians[i - settingMaxFrames].get();
      fh->flagged_for_marginalization = true;
    }
    return;
  }

  int flagged = 0;
  /// marginalize all frames that have not enough points.
  for (int i = 0; i < (int)frame_hessians.size(); i++) {
    FrameHessian *fh = frame_hessians[i].get();
    int in = fh->point_hessians.size() + fh->immature_points.size();
    int out =
        fh->point_hessians_marginalized.size() + fh->point_hessians_out.size();

    Vec2 ref_to_fh = AffLight::FromToVecExposure(
        frame_hessians.back()->exposure, fh->exposure,
        frame_hessians.back()->GetAffLight(), fh->GetAffLight());

    if ((in < settingMinPointsRemaining * (in + out) ||
         fabs(logf((float)ref_to_fh[0])) > settingMaxLogAffFacInWindow) &&
        ((int)frame_hessians.size()) - flagged > settingMinFrames) {
      fh->flagged_for_marginalization = true;
      ++flagged;
    } else {
      // do nothing here
    }
  }

  /// marginalize one, thus cannot exceed maxFrames, cannot be the flagged one
  /// before, the kf is processed one by one
  if ((int)frame_hessians.size() - flagged >= settingMaxFrames) {
    double smallest_score = 1;
    FrameHessian *to_marginalize = 0;
    FrameHessian *latest = frame_hessians.back().get();

    for (std::unique_ptr<FrameHessian> &fh : frame_hessians) {
      if (fh->frame_id > latest->frame_id - settingMinFrameAge ||
          fh->frame_id == 0) {
        continue;
      }

      /// eqn (20) in dso
      double dist_score = 0;
      for (FrameFramePrecalc &ffh : fh->target_precalc) {
        if (ffh.target->frame_id > latest->frame_id - settingMinFrameAge + 1 ||
            ffh.target == ffh.host)
          continue;
        dist_score += 1 / (1e-5 + ffh.distance_ll);
      }
      dist_score *= -sqrtf(fh->target_precalc.back().distance_ll);

      if (dist_score < smallest_score) {
        smallest_score = dist_score;
        to_marginalize = fh.get();
      }
    }

    /// Set flag to the frame with smallest score
    to_marginalize->flagged_for_marginalization = true;
    ++flagged;
  }
}

void FullSystem::MarginalizeFrame(std::unique_ptr<FrameHessian> &frame) {
  assert(frame->point_hessians.size() == 0);

  ef->MarginalizeFrame(frame->ef_frame);

  // drop all observations of existing points in that frame.
  for (std::unique_ptr<FrameHessian> &fh : frame_hessians) {
    if (fh == frame) continue;

    for (std::unique_ptr<PointHessian> &ph : fh->point_hessians) {
      for (unsigned int i = 0; i < ph->residuals.size(); i++) {
        std::unique_ptr<PointFrameResidual> &r = ph->residuals[i];
        if (r->target == frame.get()) {
          if (ph->last_residuals[0].first == r.get())
            ph->last_residuals[0].first = nullptr;
          else if (ph->last_residuals[1].first == r.get())
            ph->last_residuals[1].first = nullptr;

          // TODO: statistics
          if (r->res_blk_spec->residual_block_id) {
            parameter_map->RemoveResidualBlockSpec(r->res_blk_spec.get());
          }

          ef->DropResidual(r->ef_residual);
          DeleteOut<PointFrameResidual>(ph->residuals, i);
          break;
        }
      }
    }
  }

  if (marginalization_res_ &&
      marginalization_res_->res_blk_spec->residual_block_id) {
    parameter_map->RemoveResidualBlockSpec(
        marginalization_res_->res_blk_spec.get());
  }

  // if (frame->se3_prior_cost_function) {
  //   parameter_map->RemoveResidualBlockSpec(frame->res_blk_spec.get());
  // }

  parameter_map->RemoveParameterBlockSpec(&frame->parameter_pose);
  parameter_map->RemoveParameterBlockSpec(&frame->parameter_ab);

  frame->shell->marginalized_at = frame_hessians.back()->shell->id;
  frame->shell->moved_by_opt = frame->W2CLeftEps().norm();

  /// delete out with order pointer 0 1 2 3 (4) 5 6 -> 0 1 2 3 5 6
  DeleteOutOrder<FrameHessian>(frame_hessians, frame);
  for (unsigned int i = 0; i < frame_hessians.size(); i++)
    frame_hessians[i]->idx = i;

  SetPreCalcValues();
  ef->SetAdjointsF(HCalib);
}

void FullSystem::FlagPointsForRemoval() {
  assert(ef->ef_indices_valid);

  std::vector<FrameHessian *> fhs_to_marg;

  {
    /// push fhs to be marginalized
    for (int i = 0; i < (int)frame_hessians.size(); i++) {
      if (frame_hessians[i]->flagged_for_marginalization)
        fhs_to_marg.push_back(frame_hessians[i].get());
    }
  }

  int flag_oob = 0, flag_in = 0, flag_marg = 0, flag_nores = 0, flag_all = 0;

  // go through all active frames
  for (std::unique_ptr<FrameHessian> &host : frame_hessians) {
    for (unsigned int i = 0; i < host->point_hessians.size(); i++) {
      ++flag_all;
      std::unique_ptr<PointHessian> &ph = host->point_hessians[i];
      PointHessian *ph_ptr = ph.get();
      if (ph_ptr == nullptr) continue;

      if (ph_ptr->idist < 0 || ph_ptr->residuals.size() == 0) {
        host->point_hessians_out.emplace_back(std::move(ph));
        ph_ptr->ef_point->state_flag = EfPointStatus::DROP;
        host->point_hessians[i] = nullptr;
        flag_nores++;
      } else if (ph_ptr->IsOOB(fhs_to_marg) ||
                 host->flagged_for_marginalization) {
        // if OOB or in the flagged frame
        // WARNING: clean up codes here
        flag_oob++;

        // FIXME: FIX HERE AFTER IDIST BUG
        if (ph_ptr->IsInlierNew()) {
          flag_in++;
          int ngoodRes = 0;
          for (std::unique_ptr<PointFrameResidual> &r : ph->residuals) {
            // NOTE: re-evaluate residuals here
            // r->ResetOOB();
            // r->Linearize(HCalib);
            // r->ef_residual->is_linearized = false;
            // r->ApplyRes(true);
            if (r->ef_residual->IsActive()) {
              r->ef_residual->FixLinearizationF(ef.get());
              ngoodRes++;
            }
          }
          if (ph_ptr->idist_hessian > settingMinIdistHMarg) {
            flag_marg++;
            ph_ptr->ef_point->state_flag = EfPointStatus::MARGINALIZE;
            host->point_hessians_marginalized.emplace_back(std::move(ph));
          } else {
            ph_ptr->ef_point->state_flag = EfPointStatus::DROP;
            host->point_hessians_out.emplace_back(std::move(ph));
          }
        } else {
          ph_ptr->ef_point->state_flag = EfPointStatus::DROP;
          host->point_hessians_out.emplace_back(std::move(ph));
        }

        LOG_ASSERT(ph_ptr->ef_point->state_flag != EfPointStatus::GOOD);
        host->point_hessians[i] = nullptr;
      }

      if (ph_ptr->ef_point->state_flag == EfPointStatus::DROP) {
        for (auto &&r : ph_ptr->residuals) {
          if (r->res_blk_spec->residual_block_id) {
            parameter_map->RemoveResidualBlockSpec(r->res_blk_spec.get());
          }
        }
        parameter_map->RemoveParameterBlockSpec(&ph_ptr->parameter_idist);
      }
    }

    for (int i = 0; i < (int)host->point_hessians.size(); i++) {
      if (host->point_hessians[i] == nullptr) {
        std::swap(host->point_hessians[i], host->point_hessians.back());
        host->point_hessians.pop_back();
        --i;
      }
    }
  }

  LOG(INFO) << ">>>>>>> flag_all: " << flag_all << " flag_oob: " << flag_oob
            << " flag_in: " << flag_in << " flag_marg: " << flag_marg
            << " flag_nores: " << flag_nores;
}

void FullSystem::Marginalization() {
  // release previous parameter block info
  marginalization_info_->ResetParameterBlockInfo();
  marginalization_info_->evaluated = false;

  // FIXME: I'd like to move this part into a class...
  std::unordered_set<ParameterBlockSpec *> parameter_block_spec_to_marginalize;

  // 0. prior
  {
    for (int i = 0; i < frame_hessians.size(); ++i) {
      if (frame_hessians[i]->flagged_for_marginalization) {
        // if (frame_hessians[i]->se3_prior_cost_function) {
        //   ParameterMap::ParameterBlockCollection parameter_block_collection =
        //       parameter_map->GetParametersFromResidual(
        //           frame_hessians[i]->res_blk_spec.get());
        //
        //   ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(
        //       frame_hessians[i]->se3_prior_cost_function.get(),
        //       frame_hessians[i]->res_blk_spec->loss_function_ptr,
        //       parameter_block_collection);
        //   marginalization_info_->AddResidualBlockInfo(residual_block_info);
        // }
        parameter_block_spec_to_marginalize.insert(
            &frame_hessians[i]->parameter_pose);
        parameter_block_spec_to_marginalize.insert(
            &frame_hessians[i]->parameter_ab);
      }
    }
    if (parameter_block_spec_to_marginalize.size() > 2) {
      LOG(WARNING) << ">./>./addr_to_be_marginalized:"
                   << parameter_block_spec_to_marginalize.size();
    }

    // 1. previous marg res
    if (marginalization_res_ &&
        marginalization_res_->res_blk_spec->residual_block_id) {
      ParameterMap::ParameterBlockCollection parameter_block_collection =
          parameter_map->GetParametersFromResidual(
              marginalization_res_->res_blk_spec.get());

      std::unique_ptr<ResidualBlockInfo> residual_block_info =
          std::make_unique<ResidualBlockInfo>(
              marginalization_res_->cost_function.get(),
              marginalization_res_->res_blk_spec->loss_function_ptr,
              parameter_block_collection);
      marginalization_info_->AddResidualBlockInfo(residual_block_info);
    }
  }

  // 2. photometric res
  // FIXME: replace the ef stuffs!!!
  bool flag_latest = false;
  int res_count = 0, out_count = 0;
  double res_sum = 0, res_raw_sum = 0;
  for (std::unique_ptr<EfFrame> &f : ef->ef_frames) {
    for (int i = 0; i < (int)f->ef_points.size(); i++) {
      EfPoint *efp = f->ef_points[i].get();
      if (efp->state_flag == EfPointStatus::MARGINALIZE) {
        for (int j = 0; j < efp->ef_residuals.size(); ++j) {
          std::unique_ptr<EfResidual> &r = efp->ef_residuals[j];
          if (!r->IsActive()) continue;
          PointFrameResidual *res = r->pfr;
          if (res->ef_residual->target_idx == (frame_hessians.size() - 1)) {
            flag_latest = true;
          }
          ++res_count;
          if (r->pfr->res_energy > 1800) {
            ++out_count;
          }
          res_sum += r->pfr->res_energy;
          res_raw_sum += r->pfr->res_raw;

          parameter_block_spec_to_marginalize.insert(
              &res->point->parameter_idist);

          ParameterMap::ParameterBlockCollection parameter_block_collection =
              parameter_map->GetParametersFromResidual(res->res_blk_spec.get());
          std::unique_ptr<ResidualBlockInfo> residual_block_info =
              std::make_unique<ResidualBlockInfo>(
                  res->cost_function.get(),
                  res->res_blk_spec->loss_function_ptr,
                  parameter_block_collection);

          marginalization_info_->AddResidualBlockInfo(residual_block_info);
          // {
          //   std::vector<double *> parameters_to_zero;
          //   auto &&it = marginalization_info_->factors.back();
          //   for (auto &&p : it->parameter_block_collection) {
          //     parameters_to_zero.emplace_back(p->GetParameters());
          //   }
          //   it->EvaluateWithParameters(parameters_to_zero.data());
          //   LOG(INFO) << "DBG: " << it->residuals.dot(it->residuals) << " "
          //             << r->pfr->res_raw;
          // }

          // if (frame_hessians[res->ef_residual->host_idx]
          //         ->flagged_for_marginalization) {
          //   drop_set = std::vector<int>{2, 3};
          // } else if (frame_hessians[res->ef_residual->target_idx]
          //                ->flagged_for_marginalization) {
          //   drop_set = std::vector<int>{0, 1};
          // }
          //
          // drop_set.push_back(4);
          //
          // if (!drop_set.empty()) {
          //   ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(
          //       homo_factor, NULL,
          //       std::vector<double *>{res->target->parameter_pose.parameters,
          //                             res->target->parameter_ab.parameters,
          //                             res->host->parameter_pose.parameters,
          //                             res->host->parameter_ab.parameters,
          //                             res->point->parameter_idist.parameters},
          //       drop_set, "p");
          //   // LOG(INFO) <<
          //   // parameter_poses_[res->ef_residual->target_idx].data();
          //   marginalization_info_->AddResidualBlockInfo(residual_block_info);
          // }
        }
      }
    }
  }

  LOG(INFO) << out_count << "/" << res_count;
  LOG(INFO) << "sqrt res_sum: " << sqrt(res_sum / res_count / 8.0);
  LOG(INFO) << "sqrt res_raw_sum: " << sqrt(res_raw_sum / res_count / 8.0);

  DLOG(INFO) << (flag_latest ? "flag_latest" : "not flag_latest");

  // new parameter block info
  timing::Timer timer_pre_marg("PreMarginalize");
  marginalization_info_->PreMarginalize();
  timer_pre_marg.Stop();

  timing::Timer timer_marg("Marginalize");
  marginalization_info_->Marginalize(parameter_block_spec_to_marginalize);
  timer_marg.Stop();

  std::unordered_map<long, double *> addr_shift;
  int idx = 0;
  for (int i = 0; i < frame_hessians.size(); ++i) {
    if (frame_hessians[i]->flagged_for_marginalization) continue;
    addr_shift[reinterpret_cast<long>(
        frame_hessians[i]->parameter_pose.parameters)] =
        frame_hessians[i]->parameter_pose.parameters;
    addr_shift[reinterpret_cast<long>(
        frame_hessians[i]->parameter_ab.parameters)] =
        frame_hessians[i]->parameter_ab.parameters;
    DLOG(INFO) << "i, idx: " << i << ", " << idx;
    ++idx;
  }

  // copy into new keep (reserved) parameter block info
  marginalization_info_->GetParameterBlockCollection();
  marginalization_info_->evaluated = false;
}

}  // namespace dsl