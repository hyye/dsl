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

#ifndef DSL_PARAMETER_MAP_H_
#define DSL_PARAMETER_MAP_H_

#include <ceres/ceres.h>
#include <sophus/se3.hpp>
#include <unordered_set>

namespace dsl {

struct ParameterBlockSpec {
  bool IsFixed() const { return fixed_; }
  void SetFixed() { fixed_ = true; }

  virtual ~ParameterBlockSpec() {}

  virtual int GetDimension() const = 0;
  virtual int GetMinimalDimension() const = 0;

  virtual double* GetParameters() = 0;
  virtual const double* GetParameters() const = 0;

  virtual void SetLinearizationPoint() = 0;

  virtual double* GetParametersToZero() = 0;
  virtual const double* GetParametersToZero() const = 0;

  virtual std::string PrintParameters() const = 0;
  virtual std::string PrintParameters0() const = 0;

  virtual void Backup() = 0;

 private:
  bool fixed_ = false;
};

struct ResidualBlockSpec {
  ::ceres::ResidualBlockId residual_block_id = 0;
  ::ceres::LossFunction* loss_function_ptr = 0;
};

template <int Dim, int MinDim, class T>
struct ParameterBlockSizedSpec : ParameterBlockSpec {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  const int Dimension = Dim;
  const int MinimalDimension = MinDim;
  double parameters[Dim];
  double parameters_backup[Dim];
  double parameters0[Dim];

  virtual int GetDimension() const { return Dimension; }
  virtual int GetMinimalDimension() const { return MinimalDimension; }

  virtual double* GetParameters() { return parameters; }

  virtual const double* GetParameters() const { return parameters; }

  virtual double* GetParametersToZero() { return parameters0; }

  virtual const double* GetParametersToZero() const { return parameters0; }

  virtual std::string PrintParameters() const {
    std::stringstream ss;
    for (int i = 0; i < Dimension; ++i) {
      ss << parameters[i] << " ";
    }
    return ss.str();
  }

  virtual std::string PrintParameters0() const {
    std::stringstream ss;
    for (int i = 0; i < Dimension; ++i) {
      ss << parameters0[i] << " ";
    }
    return ss.str();
  }

  typedef T parameter_t;
  virtual void SetEstimate(const parameter_t& estimate) = 0;
  virtual parameter_t GetEstimate() const = 0;
  virtual void SetParameters(const double* _parameters) {
    if (_parameters != 0) {
      memcpy(parameters, _parameters, Dimension * sizeof(double));
      if (!IsFixed()) {
        memcpy(parameters0, parameters, sizeof(double) * Dimension);
      }
    } else {
      std::cerr << "NULL PTR!!!" << std::endl;
    }
  }

  virtual void SetLinearizationPoint() {
    memcpy(parameters0, parameters, sizeof(double) * Dimension);
    SetFixed();
  }

  virtual void Backup() {
    memcpy(parameters_backup, parameters, sizeof(double) * Dimension);
  }

  virtual parameter_t GetEstimateBackup() const = 0;
  virtual parameter_t GetLinearizedEstimate() const = 0;
};

struct PoseParameterBlockSpec
    : ParameterBlockSizedSpec<6, 6, Eigen::Matrix<double, 6, 1>> {
  virtual void SetEstimate(const Eigen::Matrix<double, 6, 1>& estimate);
  virtual Eigen::Matrix<double, 6, 1> GetEstimate() const;
  virtual Eigen::Matrix<double, 6, 1> GetEstimateBackup() const;
  virtual Eigen::Matrix<double, 6, 1> GetLinearizedEstimate() const;
};

struct AbParameterBlockSpec : ParameterBlockSizedSpec<2, 2, Eigen::Vector2d> {
  virtual void SetEstimate(const Eigen::Vector2d& estimate);
  virtual Eigen::Vector2d GetEstimate() const;
  virtual Eigen::Vector2d GetEstimateBackup() const;
  virtual Eigen::Vector2d GetLinearizedEstimate() const;
};

struct IdistParameterBlockSpec : ParameterBlockSizedSpec<1, 1, double> {
  virtual void SetEstimate(const double& estimate);
  virtual double GetEstimate() const;
  virtual double GetEstimateBackup() const;
  virtual double GetLinearizedEstimate() const;
};

struct ParameterMap {
  ParameterMap();
  bool ParameterBlockSpecExists(ParameterBlockSpec* parameter_block_spec) {
    return !(parameter_block_spec_set.find(parameter_block_spec) ==
             parameter_block_spec_set.end());
  }

  bool ResidualBlockSpecExists(ResidualBlockSpec* residual_block_spec) {
    return !(residual_block_spec_set.find(residual_block_spec) ==
             residual_block_spec_set.end());
  }

  typedef std::vector<ParameterBlockSpec*> ParameterBlockCollection;
  typedef std::vector<ResidualBlockSpec*> ResidualBlockCollection;
  typedef std::unordered_map<ResidualBlockSpec*, ParameterBlockCollection>
      ResidualBlockSpecToParameterBlockCollectionMap;
  typedef std::unordered_multimap<ParameterBlockSpec*, ResidualBlockSpec*>
      ParameterBlockToResidualBlockSpecMultimap;
  typedef std::unordered_set<ParameterBlockSpec*> ParameterBlockSpecSet;
  typedef std::unordered_set<ResidualBlockSpec*> ResidualBlockSpecSet;

  ResidualBlockCollection GetResidualsFromParameterBlock(
      ParameterBlockSpec* parameter_block_spec) const;

  ParameterBlockCollection GetParametersFromResidual(
      ResidualBlockSpec* residual_block_spec) const;

  bool AddParameterBlockSpec(
      ParameterBlockSpec* parameter_block_spec,
      ::ceres::LocalParameterization* local_parameterization = nullptr,
      const int group = -1);
  bool RemoveParameterBlockSpec(ParameterBlockSpec* parameter_block_spec);
  bool AddResidualBlockSpec(ResidualBlockSpec* res_blk_spec,
                            ::ceres::CostFunction* cost_function,
                            ::ceres::LossFunction* loss_function,
                            ParameterBlockCollection& parameter_block_specs);
  bool RemoveResidualBlockSpec(ResidualBlockSpec* residual_block_spec);

  void SolveProblem();

  ParameterBlockSpecSet parameter_block_spec_set;
  ResidualBlockSpecSet residual_block_spec_set;
  ResidualBlockSpecToParameterBlockCollectionMap
      residual_block_spec_to_parameter_block_collection_map;
  ParameterBlockToResidualBlockSpecMultimap
      parameter_block_to_residual_block_spec_multimap;

  std::unique_ptr<ceres::Problem> problem;
  ::ceres::Solver::Options options;
  ::ceres::Solver::Summary summary;
  ceres::ParameterBlockOrdering* ordering = nullptr;
};

}  // namespace dsl

#endif  // DSL_PARAMETER_MAP_H_
