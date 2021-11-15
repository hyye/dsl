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
// Created by hyye on 12/4/19.
//

#ifndef DSL_MARGINALIZATION_FACTOR_H_
#define DSL_MARGINALIZATION_FACTOR_H_

#include <ceres/ceres.h>
#include "dsl_common.h"
#include "parameter_map.h"
#include "util/num_type.h"

namespace dsl {

struct ResidualBlockInfo {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  ResidualBlockInfo(ceres::CostFunction *_cost_function,
                    ceres::LossFunction *_loss_function,
                    std::vector<double *> _parameter_blocks,
                    std::vector<int> _drop_set,
                    std::string _cost_function_name = "")
      : cost_function(_cost_function),
        loss_function(_loss_function),
        parameter_blocks(_parameter_blocks),
        drop_set(_drop_set),
        cost_function_name(_cost_function_name) {}
  ResidualBlockInfo(
      ceres::CostFunction *_cost_function, ceres::LossFunction *_loss_function,
      ParameterMap::ParameterBlockCollection _parameter_collection)
      : cost_function(_cost_function),
        loss_function(_loss_function),
        parameter_block_collection(_parameter_collection) {}

  void Evaluate();
  void EvaluateWithParameters(double const *const *parameters);

  ceres::CostFunction *cost_function;
  ceres::LossFunction *loss_function;
  std::vector<double *> parameter_blocks;
  std::vector<int> drop_set;

  ParameterMap::ParameterBlockCollection parameter_block_collection;

  std::string cost_function_name;

  double **raw_jacobians;
  std::vector<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      jacobians;
  Eigen::VectorXd residuals;

  // WARNING: deprecated?
  int LocalSize(int size) { return size == 7 ? 6 : size; }
};

struct ThreadsStruct {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  std::vector<ResidualBlockInfo *> sub_factors;
  Eigen::MatrixXd A;
  Eigen::VectorXd b;
  // TODO: avoid copy, using reference
  std::unordered_map<ParameterBlockSpec *, int> parameter_block_spec_pos_map;
};

class MarginalizationInfo {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  ~MarginalizationInfo();
  int LocalSize(int size) const;
  int GlobalSize(int size) const;
  void AddResidualBlockInfo(
      std::unique_ptr<ResidualBlockInfo> &residual_block_info);
  void PreMarginalize();
  void Marginalize(std::unordered_set<ParameterBlockSpec *>
                       &parameter_block_spec_to_marginalize);

  void ResetParameterBlockInfo();

  std::vector<ParameterBlockSpec *> GetParameterBlockCollection();

  std::vector<std::unique_ptr<ResidualBlockInfo>> factors;
  int m, n, mp, mf;
  int sum_block_size;

  // ParameterBlockSpec in the optimization route, i.e., parameter_*s
  std::unordered_set<ParameterBlockSpec *> parameter_block_spec_set;
  std::unordered_map<ParameterBlockSpec *, int> parameter_block_spec_pos_map;
  std::vector<ParameterBlockSpec *> keep_block_collection;
  std::unordered_map<ParameterBlockSpec *, int> keep_block_spec_pos_map;

  Eigen::MatrixXd linearized_jacobians;
  Eigen::VectorXd linearized_residuals;
  const double eps = 1e-8;

  bool valid = false;
  bool evaluated = false;
};

class MarginalizationFactor : public ceres::CostFunction {
 public:
  MarginalizationFactor(MarginalizationInfo *_marginalization_info);
  virtual bool Evaluate(double const *const *parameters, double *residuals,
                        double **jacobians) const;

  MarginalizationInfo *marginalization_info;
};

}  // namespace dsl

#endif  // DSL_MARGINALIZATION_FACTOR_H_
