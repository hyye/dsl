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
// Created by hyye on 2/11/20.
//

#include "full_system/optimization_call_back.h"

namespace dsl {

void OptimizationCallBack::Backup() {
  for (auto &&fh : full_system_.frame_hessians) {
    fh->parameter_pose.Backup();
    fh->parameter_ab.Backup();
    for (auto &&ph : fh->point_hessians) {
      ph->parameter_idist.Backup();
    }
  }
}

bool OptimizationCallBack::CanBreak() {
  double delta_a = 0;
  double delta_b = 0;
  double delta_trans = 0;
  double delta_rot = 0;
  double mean_idist = 0;
  double num_points = 0;

  int num_kf = (int)full_system_.frame_hessians.size();

  for (auto &&fh : full_system_.frame_hessians) {
    Eigen::Matrix<double, 6, 1> delta_se3 =
        (SE3::exp(fh->parameter_pose.GetEstimate()) *
         SE3::exp(fh->parameter_pose.GetEstimateBackup()).inverse())
            .log();
    Eigen::Matrix<double, 2, 1> delta_ab =
        fh->parameter_ab.GetEstimate() - fh->parameter_ab.GetEstimateBackup();
    delta_trans += delta_se3.segment<3>(0).squaredNorm();
    delta_rot += delta_se3.segment<3>(3).squaredNorm();
    delta_a += delta_ab.x() * delta_ab.x();
    delta_b += delta_ab.y() * delta_ab.y();
    for (auto &&ph : fh->point_hessians) {
      mean_idist += ph->parameter_idist.GetEstimateBackup();
      ++num_points;
    }
  }

  delta_a /= num_kf;
  delta_b /= num_kf;
  delta_trans /= num_kf;
  delta_rot /= num_kf;
  mean_idist /= num_points;

  if (settigNoScalingAtOpt) {
    delta_a /= SCALE_A * SCALE_A;
    delta_b /= SCALE_B * SCALE_B;
  }

  char buff[100];
  snprintf(buff, sizeof(buff),
           "STEPS: A %.1f; B %.1f; R %.1f; T %.1f. idist: %.1f \t",
           sqrt(delta_a) / 0.0006, sqrt(delta_b) / 0.00006,
           sqrt(delta_rot) / 0.00006, sqrt(delta_trans) * mean_idist / 0.00006,
           mean_idist);
  std::string buffAsStdStr = buff;
  LOG(INFO) << buffAsStdStr;

  bool converged =
      sqrt(delta_a) < 0.0006 &&     // affine light a change
      sqrt(delta_b) < 0.00006 &&    // affine light b change
      sqrt(delta_rot) < 0.00006 &&  // transformation R change
      sqrt(delta_trans) * mean_idist <
          0.00006;  // transformation T change respet to inverse depths

  return converged;
}

ceres::CallbackReturnType OptimizationCallBack::operator()(
    const ceres::IterationSummary &summary) {
  LOG(INFO) << "summary.cost @" << summary.iteration << ": " << summary.cost
            << " step: "
            << (summary.step_is_successful ? "successful" : "not successful");
  full_system_.SetNewFrameEnergyTh();

  std::stringstream ssab, ssrt;
  auto VecToStr = [](const Eigen::MatrixXd &m) {
    std::stringstream ss;
    ss << std::setprecision(4);
    for (int i = 0; i < m.size(); ++i) {
      ss << m(i) << " ";
    }
    return ss.str();
  };
  for (int i = 0; i < full_system_.frame_hessians.size(); ++i) {
    auto &&fh = full_system_.frame_hessians[i];
    // ssab << " stepAB[" << i
    //      << "]: " << VecToStr(fh->parameter_ab.GetEstimate() - abs_[i]);
    // ssrt << " stepRT[" << i << "]: "
    //      << VecToStr((SE3::exp(fh->parameter_pose.GetEstimate()) *
    //                   SE3::exp(poses_[i]).inverse())
    //                      .log());

    // poses_[i] = fh->parameter_pose.GetEstimate();
    // abs_[i] = fh->parameter_ab.GetEstimate();
  }
  // DLOG(INFO) << ssab.str() << std::endl << ssrt.str();

  bool can_break = false;

  if (summary.iteration > 0 && summary.step_is_successful) {
    timing::Timer("opt/can_break");
    can_break = this->CanBreak();
  }

  if (summary.step_is_successful) {
    full_system_.ef->ef_adjoints_valid = false;
    full_system_.ef->ef_indices_valid = false;
    full_system_.ef->SetAdjointsF(full_system_.HCalib);
    for (auto &&r : full_system_.active_residuals) {
      if (r->res_state == ResState::IN) {
        r->Jd = r->Jd_backup;
        r->center_projected_to = r->center_projected_to_backup;

        if (r->idist_converged) {
          r->point->convereged_ph_idist = true;
          r->ph_idist = r->ph_idist_backup;
        }
      }
    }
    this->Backup();
  }

  if (can_break && summary.iteration >= settingMinOptIterations) {
    return ceres::CallbackReturnType::SOLVER_TERMINATE_SUCCESSFULLY;
  }

  return ceres::CallbackReturnType::SOLVER_CONTINUE;
}

}  // namespace dsl