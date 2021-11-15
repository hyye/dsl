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
// Created by hyye on 7/14/20.
//

#include "relocalization/converter.h"
#include "relocalization/reloc_optimization/optimizer.h"
#include "relocalization/reloc_optimization/reprojection_functor.h"
#include "optimization/se3_local_parameterization.h"
#include "optimization/parameter_map.h"

namespace dsl::relocalization {

// require pFrame->mvpMapPoints to optimize
int Optimizer::PoseOptimization(Frame *pFrame) {
  int nInitialCorrespondences = 0;
  // Set MapPoint vertices
  const int N = pFrame->N;
  const float deltaMono = sqrt(5.991 * 2);  // huber loss

  std::vector<ReprojectionFunctor *> vpCostFunction;
  std::vector<size_t> vnIndexCostFunction;
  std::vector<::ceres::ResidualBlockId> vnResidualBlockId;
  vpCostFunction.reserve(N);
  vnIndexCostFunction.reserve(N);
  vnResidualBlockId.reserve(N);

  // Tcw
  PoseParameterBlockSpec pose_parameter;
  ceres::Problem problem;
  SE3LocalParameterization *localParameterization = new SE3LocalParameterization();
  // ceres::HuberLoss *loss_function = new ceres::HuberLoss(deltaMono);
  ceres::LossFunction *loss_function = new ceres::CauchyLoss(deltaMono);

  problem.AddParameterBlock(pose_parameter.parameters, pose_parameter.Dimension, localParameterization);

  for (int i = 0; i < N; i++) {
    MapPoint *pMP = pFrame->mvpMapPoints[i];
    if (pMP) {
      nInitialCorrespondences++;
      pFrame->mvbOutlier[i] = false;

      Eigen::Matrix<double, 2, 1> obs;
      const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
      const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
      obs << kpUn.pt.x, kpUn.pt.y;
      cv::Mat Xw = pMP->GetWorldPos();
      ReprojectionFunctor *cost_function = new ReprojectionFunctor();
      cost_function->measurement = obs;
      cost_function->SetInformation(Eigen::Matrix2d::Identity() * invSigma2);
      cost_function->Xw = Converter::toVector3d(Xw);

      cost_function->fx = pFrame->fx;
      cost_function->fy = pFrame->fy;
      cost_function->cx = pFrame->cx;
      cost_function->cy = pFrame->cy;

      ::ceres::ResidualBlockId
          res_id = problem.AddResidualBlock(cost_function, loss_function, pose_parameter.parameters);

      vpCostFunction.push_back(cost_function);
      vnIndexCostFunction.push_back(i);
      vnResidualBlockId.push_back(res_id);
    }
  }

  if (nInitialCorrespondences < 3)
    return 0;

  // we follow the ORB's optimization
  const float chi2Mono = 5.991 * 2;

  int nBad = 0;
  std::vector<double *> vpParamters;
  vpParamters.push_back(pose_parameter.parameters);
  for (size_t it = 0; it < 4; it++) {
    // TODO: optimize me
    pose_parameter.SetParameters(Converter::toSE3Quat(pFrame->mTcw_pnp).log().data());

    // Run the solver!
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    options.max_linear_solver_iterations = 10;
    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);

    SE3 Tcw_est = SE3::exp(pose_parameter.GetEstimate());

    nBad = 0;
    // filter bad points

    for (int i = 0; i < vpCostFunction.size(); ++i) {
      ReprojectionFunctor *cost_function = vpCostFunction[i];
      cost_function->ComputeError(vpParamters.data());
      const size_t idx = vnIndexCostFunction[i];

      if (pFrame->mvbOutlier[idx]) {
        // error is updated above
      }

      const float chi2 = cost_function->GetChi2();

      if (chi2 > chi2Mono) {
        pFrame->mvbOutlier[idx] = true;
        cost_function->SetFixed(true, NULL);
        nBad++;
      } else {
        pFrame->mvbOutlier[idx] = false;
        cost_function->SetFixed(false, NULL);
      }

      if (it == 2) {
        ReprojectionFunctor *cost_function_new = new ReprojectionFunctor(*cost_function);

        // remove old
        ::ceres::ResidualBlockId res_id = vnResidualBlockId[i];
        problem.RemoveResidualBlock(res_id);

        // add new
        ::ceres::ResidualBlockId
            res_id_new = problem.AddResidualBlock(cost_function_new, NULL, pose_parameter.parameters);
        vpCostFunction[i] = cost_function_new;
        vnResidualBlockId[i] = res_id_new;
      }
    }

    if (nInitialCorrespondences - nBad < 10) break;

    // Tcw_est = SE3::exp(pose_parameter.GetEstimate());
    // cv::Mat Tcw = Converter::toCvMat(Tcw_est);
    // pFrame->mTcw_pnp = Tcw;
  }

  SE3 Tcw_est = SE3::exp(pose_parameter.GetEstimate());
  cv::Mat Tcw = Converter::toCvMat(Tcw_est);
  pFrame->mTcw_pnp = Tcw;
  // pFrame->SetPose(Tcw);

  return nInitialCorrespondences - nBad;
}

} // namespace dsl::relocalization
