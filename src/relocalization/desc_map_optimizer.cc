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
// Created by hyye on 8/24/20.
//

#include "util/global_calib.h"
#include "relocalization/desc_map.h"
#include "relocalization/converter.h"
#include "relocalization/feature_matcher.h"
#include "relocalization/sp_matcher.h"

#include "relocalization/reloc_optimization/optimizer.h"
#include "relocalization/reloc_optimization/surfel_reprojection_functor.h"
#include "optimization/se3_local_parameterization.h"
#include "optimization/parameter_map.h"

#include "util/timing.h"
#include <opencv2/calib3d.hpp>

#include <cmath>
#include <algorithm>

#include <boost/iterator/counting_iterator.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <boost/serialization/unique_ptr.hpp>

#include <queue>
#include <tool/dataset_converter.h>

#include "fmt/core.h"
#include "fmt/color.h"

namespace dsl::relocalization {

const double MAX_ERROR = sqrt(5.991 * 1 / 460.0);

unsigned long GetObsFirstKFId(const std::map<Frame *, size_t> &obs) {
  unsigned long firstKFId = std::numeric_limits<unsigned long>::max();
  for (auto &&pF_featId : obs) {
    Frame *pF = pF_featId.first;
    if (firstKFId > pF->mnId) firstKFId = pF->mnId;
  }
  LOG_ASSERT(firstKFId != 0 && firstKFId != std::numeric_limits<unsigned long>::max()) << firstKFId;
  return firstKFId;
}

bool AddParameterToProblem(ceres::Problem &problem,
                           std::map<unsigned long, std::unique_ptr<PoseParameterBlockSpec>> &frameId_poseParameter,
                           unsigned long frameId,
                           ceres::LocalParameterization *localParameterization,
                           const double *data) {
  if (!frameId_poseParameter.count(frameId)) {
    frameId_poseParameter[frameId] = std::make_unique<PoseParameterBlockSpec>();
    problem.AddParameterBlock(frameId_poseParameter[frameId]->parameters,
                              frameId_poseParameter[frameId]->Dimension,
                              localParameterization);
    frameId_poseParameter[frameId]->SetParameters(data);
    return true;
  }
  return false;
}

void DescMap::DoOptimization(bool run_opt) {
  timing::Timer timer("map_opt");
  const double deltaMono = MAX_ERROR;  // huber loss
  ceres::Problem problem;
  SE3LocalParameterization *localParameterization = new SE3LocalParameterization();
  // ceres::HuberLoss *loss_function = new ceres::HuberLoss(deltaMono);
  ceres::LossFunction *loss_function = new ceres::CauchyLoss(deltaMono);

  std::map<unsigned long, std::unique_ptr<PoseParameterBlockSpec>> frameId_poseParameter;
  Eigen::Matrix3d Kinv = KiG[0].cast<double>();

  std::vector<SurfelReprojectionFunctor *> vpCostFunction;
  std::vector<::ceres::ResidualBlockId> vnResidualBlockId;
  std::vector<std::tuple<unsigned long, unsigned long, size_t, unsigned long, size_t>> vtFrameIdFeatId;
  std::vector<std::pair<double *, double *>> vParamaters;

  // Formulate the optimization problem
  for (auto &&id_pMP: all_map_points) {
    unsigned long mpId = id_pMP.first;
    MapPoint *pMP = id_pMP.second.get();
    std::map<Frame *, size_t> obs = pMP->mObservations;
    unsigned long global_idx = pMP->idx_in_surfel_map;
    unsigned long firstKFId = GetObsFirstKFId(obs);

    Frame *pFirstKF = all_keyframes.at(firstKFId).get();
    size_t featIdFirstKF = obs[pFirstKF];

    AddParameterToProblem(problem,
                          frameId_poseParameter,
                          firstKFId,
                          localParameterization,
                          Converter::toSE3Quat(pFirstKF->GetPoseInverse()).log().data());

    cv::Point3f vertex = GetGlobalPoint(global_idx);
    cv::Point3f normal = GetGlobalNormal(global_idx);

    Eigen::Vector3d n_w(normal.x, normal.y, normal.z);
    Eigen::Vector3d v_w(vertex.x, vertex.y, vertex.z);
    double d_w = -n_w.dot(v_w);

    for (auto&&[pCovKF, featIdCovKF] : obs) {
      if (pCovKF->mnId == firstKFId) continue;
      LOG_ASSERT(pCovKF->mnId > firstKFId);
      // LOG(INFO) << firstKFId << ": " << pCovKF->mnId;
      AddParameterToProblem(problem,
                            frameId_poseParameter,
                            pCovKF->mnId,
                            localParameterization,
                            Converter::toSE3Quat(pCovKF->GetPoseInverse()).log().data());

      Eigen::Vector3d x0 =
          Kinv * Eigen::Vector3d(pFirstKF->mvKeysUn[featIdFirstKF].pt.x, pFirstKF->mvKeysUn[featIdFirstKF].pt.y, 1);
      Eigen::Vector3d x1 =
          Kinv * Eigen::Vector3d(pCovKF->mvKeysUn[featIdCovKF].pt.x, pCovKF->mvKeysUn[featIdCovKF].pt.y, 1);

      SurfelReprojectionFunctor *cost_function = new SurfelReprojectionFunctor(x0, x1, n_w, d_w);

      ::ceres::ResidualBlockId
          res_id = problem.AddResidualBlock(cost_function, loss_function, frameId_poseParameter[firstKFId]->parameters,
                                            frameId_poseParameter[pCovKF->mnId]->parameters);

      vpCostFunction.push_back(cost_function);
      vnResidualBlockId.push_back(res_id);
      vtFrameIdFeatId.emplace_back(mpId, firstKFId, featIdFirstKF, pCovKF->mnId, featIdCovKF);

      vParamaters.emplace_back(frameId_poseParameter[firstKFId]->parameters,
                               frameId_poseParameter[pCovKF->mnId]->parameters);
      // LOG(INFO) << vParamaters.back()[1][0] << " " << frameId_poseParameter[pCovKF->mnId]->parameters[0];
    }
  }

  LOG(INFO) << frameId_poseParameter.size();

  // Run optimization
  // Run the solver!
  if (run_opt) {
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    // options.max_linear_solver_iterations = 10;
    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);
  }

  // for (int i = 0; i < vpCostFunction.size(); ++i) {
  //   SurfelReprojectionFunctor *cost_function = vpCostFunction[i];
  //   std::pair<double *, double *> params = vParamaters[i];
  //   std::vector<double *> vec_params;
  //   vec_params.push_back(params.first);
  //   vec_params.push_back(params.second);
  //
  //   auto[mpId, firstKFId, featIdFirstKF, CovKFId, featIdCovKF] = vtFrameIdFeatId[i];
  //
  //   // LOG(INFO) << params.second[0] << " " << frameId_poseParameter[CovKFId]->parameters[0];
  //   cost_function->ComputeError(vec_params.data());
  //
  //   if (cost_function->error > MAX_ERROR) {
  //     LOG(INFO) << fmt::format("OUTLIER: {}, {}, {}, {}, {}",
  //                              cost_function->error,
  //                              firstKFId,
  //                              featIdFirstKF,
  //                              CovKFId,
  //                              featIdCovKF);
  //     if(all_map_points.count(mpId)) {
  //       all_map_points.at(mpId)->EraseObservation(all_keyframes[CovKFId].get());
  //     }
  //   }
  // }

  // Update keyframe pos
  for (auto&&[frame_id, pKF]: all_keyframes) {
    SE3 T_wc_opt = SE3::exp(frameId_poseParameter[frame_id]->GetEstimate());
    pKF->SetPose(Converter::toCvMat(T_wc_opt.inverse()));
  }

  // Update map point pos
  for (auto &&id_pMP: all_map_points) {
    MapPoint *pMP = id_pMP.second.get();

    std::map<Frame *, size_t> obs = id_pMP.second->mObservations;
    unsigned long firstKFId = GetObsFirstKFId(obs);
    LOG_ASSERT(all_keyframes.count(firstKFId));
    Frame *pFirstKF = all_keyframes.at(firstKFId).get();
    size_t featIdFirstKF = obs[pFirstKF];

    Eigen::Vector3d x0 =
        Kinv * Eigen::Vector3d(pFirstKF->mvKeysUn[featIdFirstKF].pt.x, pFirstKF->mvKeysUn[featIdFirstKF].pt.y, 1);

    SE3 T_wc = Converter::toSE3Quat(pFirstKF->GetPoseInverse());

    unsigned long global_idx = pMP->idx_in_surfel_map;
    cv::Point3f vertex = GetGlobalPoint(global_idx);
    cv::Point3f normal = GetGlobalNormal(global_idx);

    Eigen::Vector3d n_w(normal.x, normal.y, normal.z);
    Eigen::Vector3d v_w(vertex.x, vertex.y, vertex.z);
    double d_w = -n_w.dot(v_w);

    double rho = -(n_w.dot(T_wc.rotationMatrix() * x0)) / (n_w.dot(T_wc.translation()) + d_w);
    Eigen::Vector3d x_w = T_wc * (x0 / rho);
    pMP->mWorldPos = Converter::toCvMat(x_w);
  }
  timer.Stop();

  std::ofstream opt_file;
  opt_file.open("/tmp/optimized.tum");

  for (auto&&[frame_id, pKF]: all_keyframes) {
    opt_file << ToTum(pKF->mTimeStamp, Converter::toSE3Quat(pKF->GetPoseInverse()));
  }

  opt_file.close();
  LOG(INFO) << timing::Timing::Print();

}

}