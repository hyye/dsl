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
// Created by hyye on 1/7/20.
//

#include "optimization/parameter_map.h"
#include "dsl_common.h"

namespace dsl {

// TODO: duplicate
void PoseParameterBlockSpec::SetEstimate(
    const Eigen::Matrix<double, 6, 1> &estimate) {
  Eigen::Map<Eigen::Matrix<double, 6, 1>> internal_est(parameters);
  internal_est = estimate;
  if (!IsFixed()) {
    memcpy(parameters0, parameters, sizeof(double) * Dimension);
  }
}

Eigen::Matrix<double, 6, 1> PoseParameterBlockSpec::GetEstimate() const {
  Eigen::Map<const Eigen::Matrix<double, 6, 1>> estimate(parameters);
  return estimate;
}

Eigen::Matrix<double, 6, 1> PoseParameterBlockSpec::GetEstimateBackup() const {
  Eigen::Map<const Eigen::Matrix<double, 6, 1>> estimate(parameters_backup);
  return estimate;
}

Eigen::Matrix<double, 6, 1> PoseParameterBlockSpec::GetLinearizedEstimate()
    const {
  if (IsFixed()) {
    Eigen::Map<const Eigen::Matrix<double, 6, 1>> estimate0(parameters0);
    return estimate0;
  } else {
    return GetEstimate();
  }
}

void AbParameterBlockSpec::SetEstimate(const Eigen::Vector2d &estimate) {
  Eigen::Map<Eigen::Vector2d> internal_est(parameters);
  internal_est = estimate;
  if (!IsFixed()) {
    memcpy(parameters0, parameters, sizeof(double) * Dimension);
  }
}
Eigen::Vector2d AbParameterBlockSpec::GetEstimate() const {
  Eigen::Map<const Eigen::Vector2d> estimate(parameters);
  return estimate;
}

Eigen::Vector2d AbParameterBlockSpec::GetEstimateBackup() const {
  Eigen::Map<const Eigen::Vector2d> estimate(parameters_backup);
  return estimate;
}

Eigen::Vector2d AbParameterBlockSpec::GetLinearizedEstimate() const {
  if (IsFixed()) {
    Eigen::Map<const Eigen::Vector2d> estimate0(parameters0);
    return estimate0;
  } else {
    return GetEstimate();
  }
}

void IdistParameterBlockSpec::SetEstimate(const double &estimate) {
  parameters[0] = estimate;
  if (!IsFixed()) {
    memcpy(parameters0, parameters, sizeof(double) * Dimension);
  }
}
double IdistParameterBlockSpec::GetEstimate() const { return parameters[0]; }
double IdistParameterBlockSpec::GetEstimateBackup() const {
  return parameters_backup[0];
}

double IdistParameterBlockSpec::GetLinearizedEstimate() const {
  if (IsFixed()) {
    return parameters0[0];
  } else {
    return GetEstimate();
  }
}

ParameterMap::ParameterMap() {
  ::ceres::Problem::Options problem_options;
  problem_options.local_parameterization_ownership =
      ::ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;
  problem_options.loss_function_ownership =
      ::ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;
  problem_options.cost_function_ownership =
      ::ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;
  problem_options.enable_fast_removal = true;
  problem = std::make_unique<::ceres::Problem>(problem_options);
}

ParameterMap::ResidualBlockCollection
ParameterMap::GetResidualsFromParameterBlock(
    ParameterBlockSpec *parameter_block_spec) const {
  ParameterBlockToResidualBlockSpecMultimap::const_iterator it1 =
      parameter_block_to_residual_block_spec_multimap.find(
          parameter_block_spec);
  ResidualBlockCollection residual_ids;
  if (it1 != parameter_block_to_residual_block_spec_multimap.end()) {
    std::pair<ParameterBlockToResidualBlockSpecMultimap::const_iterator,
              ParameterBlockToResidualBlockSpecMultimap::const_iterator>
        range = parameter_block_to_residual_block_spec_multimap.equal_range(
            parameter_block_spec);
    for (ParameterBlockToResidualBlockSpecMultimap::const_iterator it =
             range.first;
         it != range.second; ++it) {
      residual_ids.push_back(it->second);
    }
  }
  return residual_ids;
}

ParameterMap::ParameterBlockCollection ParameterMap::GetParametersFromResidual(
    ResidualBlockSpec *residual_block_spec) const {
  ResidualBlockSpecToParameterBlockCollectionMap::const_iterator it =
      residual_block_spec_to_parameter_block_collection_map.find(
          residual_block_spec);
  if (it == residual_block_spec_to_parameter_block_collection_map.end()) {
    LOG_ASSERT(false) << " " << residual_block_spec;
    ParameterBlockCollection empty;
    return empty;  // empty vector
  }
  return it->second;
}

bool ParameterMap::AddParameterBlockSpec(
    ParameterBlockSpec *parameter_block_spec,
    ::ceres::LocalParameterization *local_parameterization, int group) {
  timing::Timer add_timer("parameter_map/add_param");
  if (ParameterBlockSpecExists(parameter_block_spec)) return false;
  parameter_block_spec_set.insert(parameter_block_spec);
  if (local_parameterization) {
    problem->AddParameterBlock(parameter_block_spec->GetParameters(),
                               parameter_block_spec->GetDimension(),
                               local_parameterization);
  } else {
    problem->AddParameterBlock(parameter_block_spec->GetParameters(),
                               parameter_block_spec->GetDimension());
  }
  if (group >= 0) {
    if (!ordering) {
      ordering = new ceres::ParameterBlockOrdering;
    }
    ordering->AddElementToGroup(parameter_block_spec->GetParameters(), group);
  }
  return true;
}
bool ParameterMap::RemoveParameterBlockSpec(
    ParameterBlockSpec *parameter_block_spec) {
  timing::Timer add_timer("parameter_map/remove_param");

  if (!ParameterBlockSpecExists(parameter_block_spec)) {
    LOG(FATAL) << "remove failed" << parameter_block_spec_set.size();
    return false;
  }

  if (ordering) {
    ordering->Remove(parameter_block_spec->GetParameters());
  }

  // const ResidualBlockCollection res_ids =
  //     GetResidualsFromParameterBlock(parameter_block_spec);
  // for (size_t i = 0; i < res_ids.size(); ++i) {
  //   RemoveResidualBlockSpec(res_ids[i]);
  // }
  problem->RemoveParameterBlock(parameter_block_spec->GetParameters());
  parameter_block_spec_set.erase(parameter_block_spec);
  return true;
}
bool ParameterMap::AddResidualBlockSpec(
    ResidualBlockSpec *res_blk_spec, ::ceres::CostFunction *cost_function,
    ::ceres::LossFunction *loss_function,
    ParameterBlockCollection &parameter_block_specs) {
  // DLOG(INFO) << "ADD: " << res_blk_spec;
  timing::Timer add_timer("parameter_map/add_res");
  if (ResidualBlockSpecExists(res_blk_spec)) return false;

  std::vector<double *> parameter_blocks;
  for (int i = 0; i < parameter_block_specs.size(); ++i) {
    parameter_blocks.push_back(parameter_block_specs[i]->GetParameters());
  }
  res_blk_spec->residual_block_id =
      problem->AddResidualBlock(cost_function, loss_function, parameter_blocks);
  res_blk_spec->loss_function_ptr = loss_function;

  residual_block_spec_set.insert(res_blk_spec);

  std::pair<ResidualBlockSpecToParameterBlockCollectionMap::iterator, bool>
      insertion = residual_block_spec_to_parameter_block_collection_map.insert(
          std::pair<ResidualBlockSpec *, ParameterBlockCollection>(
              res_blk_spec, parameter_block_specs));
  if (!insertion.second) {
    LOG_ASSERT(false) << res_blk_spec;
    return false;
  }

  // for (uint64_t parameter_id = 0; parameter_id <
  // parameter_block_specs.size();
  //      ++parameter_id) {
  //   parameter_block_to_residual_block_spec_multimap.insert(
  //       std::pair<ParameterBlockSpec *, ResidualBlockSpec*>(
  //           parameter_block_specs[parameter_id], res_id));
  // }

  return true;
}
bool ParameterMap::RemoveResidualBlockSpec(
    ResidualBlockSpec *residual_block_spec) {
  // DLOG(INFO) << "REMOVE: " << residual_block_spec;
  timing::Timer add_timer("parameter_map/remove_res");
  if (!ResidualBlockSpecExists(residual_block_spec)) return false;

  problem->RemoveResidualBlock(residual_block_spec->residual_block_id);
  residual_block_spec->residual_block_id = 0;
  timing::Timer timer1("parameter_map/remove_res/1");
  ResidualBlockSpecToParameterBlockCollectionMap::iterator it =
      residual_block_spec_to_parameter_block_collection_map.find(
          residual_block_spec);
  if (it == residual_block_spec_to_parameter_block_collection_map.end())
    return false;
  timer1.Stop();

  timing::Timer timer2("parameter_map/remove_res/2");
  // for (ParameterBlockCollection::iterator parameter_it = it->second.begin();
  //      parameter_it != it->second.end(); ++parameter_it) {
  //   ParameterBlockSpec *parameter_block_spec = *parameter_it;
  //   std::pair<ParameterBlockToResidualBlockSpecMultimap::iterator,
  //             ParameterBlockToResidualBlockSpecMultimap::iterator>
  //       range = parameter_block_to_residual_block_spec_multimap.equal_range(
  //           parameter_block_spec);
  //
  //   for (ParameterBlockToResidualBlockSpecMultimap::iterator it2 =
  //   range.first;
  //        it2 != range.second;) {
  //     if (residual_block_spec == it2->second) {
  //       // Iterator following the last removed element.
  //       it2 = parameter_block_to_residual_block_spec_multimap.erase(it2);
  //     } else {
  //       it2++;
  //     }
  //   }
  // }
  timer2.Stop();
  timing::Timer timer3("parameter_map/remove_res/3");
  residual_block_spec_to_parameter_block_collection_map.erase(it);
  residual_block_spec_set.erase(residual_block_spec);
  timer3.Stop();
  return true;
}

void ParameterMap::SolveProblem() {
  if (options.linear_solver_ordering.get() != ordering) {
    options.linear_solver_ordering.reset(ordering);
  }
  Solve(options, problem.get(), &summary);
}

}  // namespace dsl