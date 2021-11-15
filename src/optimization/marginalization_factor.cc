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

#include "optimization/marginalization_factor.h"
#include <thread>
#include "full_system/hessian_blocks.h"
#include "optimization/homo_cost_functor.h"

namespace dsl {

void ResidualBlockInfo::EvaluateWithParameters(
    const double *const *parameters) {
  residuals.resize(cost_function->num_residuals());

  std::vector<int> block_sizes = cost_function->parameter_block_sizes();
  raw_jacobians = new double *[block_sizes.size()];
  jacobians.resize(block_sizes.size());

  for (int i = 0; i < static_cast<int>(block_sizes.size()); i++) {
    jacobians[i].resize(cost_function->num_residuals(), block_sizes[i]);
    raw_jacobians[i] = jacobians[i].data();
  }
  cost_function->Evaluate(parameters, residuals.data(), raw_jacobians);

  // http://ceres-solver.org/nnls_modeling.html?highlight=loss%20function#lossfunction
  if (loss_function) {
    double residual_scaling_, alpha_sq_norm_;

    double sq_norm, rho[3];

    sq_norm = residuals.squaredNorm();
    loss_function->Evaluate(sq_norm, rho);

    double sqrt_rho1_ = sqrt(rho[1]);

    if ((sq_norm == 0.0) || (rho[2] <= 0.0)) {
      residual_scaling_ = sqrt_rho1_;
      alpha_sq_norm_ = 0.0;
    } else {
      const double D = 1.0 + 2.0 * sq_norm * rho[2] / rho[1];
      const double alpha = 1.0 - sqrt(D);
      residual_scaling_ = sqrt_rho1_ / (1 - alpha);
      alpha_sq_norm_ = alpha / sq_norm;
    }

    for (int i = 0; i < static_cast<int>(parameter_blocks.size()); i++) {
      jacobians[i] = sqrt_rho1_ * (jacobians[i] -
                                   alpha_sq_norm_ * residuals *
                                       (residuals.transpose() * jacobians[i]));
    }

    residuals *= residual_scaling_;
  }
}

// http://ceres-solver.org/nnls_modeling.html#_CPPv2N5ceres12CostFunction8EvaluateEPPCdPdPPd
void ResidualBlockInfo::Evaluate() {
  EvaluateWithParameters(parameter_blocks.data());
}

MarginalizationInfo::~MarginalizationInfo() {
  for (int i = 0; i < (int)factors.size(); i++) {
    delete[] factors[i]->raw_jacobians;

    /// in fh or ph
    // delete factors[i]->cost_function;

    factors[i].reset();
  }
}

void MarginalizationInfo::AddResidualBlockInfo(
    std::unique_ptr<ResidualBlockInfo> &residual_block_info) {
  factors.emplace_back(std::move(residual_block_info));
}

void MarginalizationInfo::PreMarginalize() {
  LOG(INFO) << "+++++++ factors.size(): " << factors.size();

  timing::Timer pre_timer("dsl/keyframe/marginalization/pre");

  int num_out = 0, res_count = 0;
  double cost_all = 0, cost_raw = 0;

  // TODO: parallel?
  for (auto &&it : factors) {
    // NOTE: parameter bug removed
    std::vector<double *> parameters_to_zero;
    std::vector<double *> parameters_new;
    std::vector<int> block_sizes = it->cost_function->parameter_block_sizes();
    LOG_ASSERT(block_sizes.size() == it->parameter_block_collection.size())
        << " " << block_sizes.size() << ":"
        << it->parameter_block_collection.size();

    HomoCostFunctor *homo_ptr =
        dynamic_cast<HomoCostFunctor *>(it->cost_function);

    for (int i = 0; i < block_sizes.size(); ++i) {
      ParameterBlockSpec *parameter_block_spec =
          it->parameter_block_collection[i];

      if (homo_ptr != nullptr && homo_ptr->pfr_->idist_converged &&
          dynamic_cast<IdistParameterBlockSpec *>(parameter_block_spec) !=
              nullptr) {
       // it should not be added.
      } else {
        parameter_block_spec_set.insert(parameter_block_spec);
      }

      if (!parameter_block_spec->IsFixed()) {
        parameter_block_spec->SetLinearizationPoint();
      }
      parameters_to_zero.emplace_back(
          parameter_block_spec->GetParametersToZero());
      // LOG(INFO) << "p: " << parameter_block_spec->PrintParameters();
      // LOG(INFO) << "p0: " << parameter_block_spec->PrintParameters0();
      parameters_new.emplace_back(parameter_block_spec->GetParameters());
    }

    it->EvaluateWithParameters(parameters_to_zero.data());
    // it->EvaluateWithParameters(parameters_new.data());

    if (homo_ptr != nullptr) {
      if (sqrt(it->residuals.dot(it->residuals) / 8) > 15) {
        // LOG(ERROR) << ">>>>>>> it->residuals.dot(it->residuals) too large: "
        //            << sqrt(it->residuals.dot(it->residuals) / 8) << "
        //            <<<<<<<";
        // it->EvaluateWithParameters(parameters_new.data());
        // LOG(ERROR) << ">>>>>>> it->residuals.dot(it->residuals) NOW: "
        //            << sqrt(it->residuals.dot(it->residuals) / 8) << "
        //            <<<<<<<";
        // it->EvaluateWithParameters(parameters_to_zero.data());
        ++num_out;
      }
      // it->EvaluateWithParameters(parameters_new.data());
      cost_all += it->residuals.dot(it->residuals);
      // it->EvaluateWithParameters(parameters_to_zero.data());
      ++res_count;
    }
  }
  LOG(INFO) << ">>>>>>> num_out / factors.size(): " << num_out << " / "
            << (int)factors.size() - 1
            << "; rmse: " << sqrt(cost_all / res_count / 8.0) << " <<<<<<<";
}

// WARNING: deprecated?
int MarginalizationInfo::LocalSize(int size) const {
  // return size == 7 ? 6 : size;
  return size;
}

int MarginalizationInfo::GlobalSize(int size) const {
  // return size == 6 ? 7 : size;
  return size;
}

void ThreadsConstructA(ThreadsStruct &threadsstruct) {
  ThreadsStruct &p = threadsstruct;
  for (auto it : p.sub_factors) {
    for (int i = 0; i < static_cast<int>(it->parameter_block_collection.size());
         i++) {
      LOG_ASSERT(p.parameter_block_spec_pos_map.find(
                     it->parameter_block_collection[i]) !=
                 p.parameter_block_spec_pos_map.end());
      int idx_i =
          p.parameter_block_spec_pos_map[it->parameter_block_collection[i]];
      int size_i = it->parameter_block_collection[i]->GetMinimalDimension();

      Eigen::MatrixXd jacobian_i = it->jacobians[i].leftCols(size_i);
      for (int j = i;
           j < static_cast<int>(it->parameter_block_collection.size()); j++) {
        int idx_j =
            p.parameter_block_spec_pos_map[it->parameter_block_collection[j]];
        int size_j = it->parameter_block_collection[j]->GetMinimalDimension();

        Eigen::MatrixXd jacobian_j = it->jacobians[j].leftCols(size_j);
        if (i == j)
          p.A.block(idx_i, idx_j, size_i, size_j) +=
              jacobian_i.transpose() * jacobian_j;
        else {
          p.A.block(idx_i, idx_j, size_i, size_j) +=
              jacobian_i.transpose() * jacobian_j;
          p.A.block(idx_j, idx_i, size_j, size_i) =
              p.A.block(idx_i, idx_j, size_i, size_j).transpose();
        }
        // LOG(INFO) << idx_i << " " << idx_j << " " << size_i << " " << size_j;
      }
      p.b.segment(idx_i, size_i) += jacobian_i.transpose() * it->residuals;
    }
  }
}

bool ParameterBlockSpecExists(
    std::unordered_set<ParameterBlockSpec *> &parameter_block_spec_set,
    ParameterBlockSpec *parameter_block_spec) {
  return !(parameter_block_spec_set.find(parameter_block_spec) ==
           parameter_block_spec_set.end());
}

void MarginalizationInfo::Marginalize(std::unordered_set<ParameterBlockSpec *> &
                                          parameter_block_spec_to_marginalize) {
  int pos = 0;
  // for (auto &it : parameter_block_idx) {
  //   it.second = pos;
  //   pos += LocalSize(parameter_block_size[it.first]);
  // }

  // for (auto &it : point_parameter_block_addr) {
  //   parameter_block_idx[it] = pos;
  //   pos += LocalSize(parameter_block_size[it]);
  // }
  // mp = pos;
  //
  // for (auto &it : frame_parameter_block_addr) {
  //   parameter_block_idx[it] = pos;
  //   pos += LocalSize(parameter_block_size[it]);
  // }
  // mf = pos - mp;
  //
  // m = pos;

  // for (const auto &it : parameter_block_size) {
  //   if (parameter_block_idx.find(it.first) == parameter_block_idx.end()) {
  //     parameter_block_idx[it.first] = pos;
  //     pos += LocalSize(it.second);
  //   }
  // }

  for (const auto &it : parameter_block_spec_set) {
    if (ParameterBlockSpecExists(parameter_block_spec_to_marginalize, it)) {
      if (dynamic_cast<IdistParameterBlockSpec *>(it) != nullptr) {
        parameter_block_spec_pos_map[it] = pos;
        pos += it->GetMinimalDimension();
      }
    }
  }
  mp = pos;

  for (const auto &it : parameter_block_spec_set) {
    if (ParameterBlockSpecExists(parameter_block_spec_to_marginalize, it)) {
      if (dynamic_cast<IdistParameterBlockSpec *>(it) == nullptr) {
        parameter_block_spec_pos_map[it] = pos;
        pos += it->GetMinimalDimension();
      }
    }
  }
  mf = pos - mp;
  m = pos;

  std::stringstream ss_out;

  for (const auto &it : parameter_block_spec_set) {
    if (!ParameterBlockSpecExists(parameter_block_spec_to_marginalize, it)) {
      parameter_block_spec_pos_map[it] = pos;
      pos += it->GetMinimalDimension();
      if (dynamic_cast<AbParameterBlockSpec *>(it) != nullptr) {
        ss_out << "ab ";
      }
      if (dynamic_cast<PoseParameterBlockSpec *>(it) != nullptr) {
        ss_out << "pose ";
      }
    }
  }
  n = pos - m;

  DLOG(INFO) << "m, n: " << m << ", " << n;
  DLOG(INFO) << "mp, mf, pos: " << mp << ", " << mf << ", " << pos;
  DLOG(INFO) << ss_out.str();

  if (m > 0) {
    valid = true;
  } else if (!valid) {
    return;
  }

  timing::Timer pile_timer("dsl/keyframe/marginalization/pile");
  Eigen::MatrixXd A(pos, pos);
  Eigen::VectorXd b(pos);
  A.setZero();
  b.setZero();

  std::thread threads[NUM_THREADS];
  ThreadsStruct threadsstruct[NUM_THREADS];
  int i = 0;
  for (auto &&it : factors) {
    threadsstruct[i].sub_factors.push_back(it.get());
    i++;
    i = i % NUM_THREADS;
  }
  for (int i = 0; i < NUM_THREADS; i++) {
    threadsstruct[i].A = Eigen::MatrixXd::Zero(pos, pos);
    threadsstruct[i].b = Eigen::VectorXd::Zero(pos);
    threadsstruct[i].parameter_block_spec_pos_map =
        parameter_block_spec_pos_map;
    threads[i] = std::thread(ThreadsConstructA, std::ref(threadsstruct[i]));
    if (!threads[i].joinable()) {
      LOG(FATAL) << "thread create error";
    }
  }
  for (int i = NUM_THREADS - 1; i >= 0; i--) {
    threads[i].join();
    A += threadsstruct[i].A;
    b += threadsstruct[i].b;
  }
  pile_timer.Stop();

  // TODO

  timing::Timer inverse_timer("dsl/keyframe/marginalization/inverse");
  {
    int rf = mf + n;
    Eigen::MatrixXd Ampmp = A.block(0, 0, mp, mp);
    Eigen::MatrixXd Ampmp_inv =
        Eigen::VectorXd((Ampmp.diagonal().array() > eps)
                            .select(Ampmp.diagonal().array().inverse(), 0))
            .asDiagonal();
    Eigen::VectorXd bmm = b.segment(0, mp);
    Eigen::MatrixXd Amr = A.block(0, mp, mp, rf);
    Eigen::MatrixXd Arm = A.block(mp, 0, rf, mp);
    Eigen::MatrixXd Arr = A.block(mp, mp, rf, rf);
    Eigen::VectorXd brr = b.segment(mp, rf);
    // LOG(INFO) << (Arm * Ampmp_inv * Amr).size();
    // LOG(INFO) << Arr.size();
    A = (Arr - Arm * Ampmp_inv * Amr).eval();
    b = (brr - Arm * Ampmp_inv * bmm).eval();

    if (mf > 0) {
      Eigen::MatrixXd Amfmf =
          0.5 * (A.block(0, 0, mf, mf) + A.block(0, 0, mf, mf).transpose());
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes_mf(Amfmf);

      Eigen::MatrixXd Amfmf_inv =
          saes_mf.eigenvectors() *
          Eigen::VectorXd(
              (saes_mf.eigenvalues().array() > eps)
                  .select(saes_mf.eigenvalues().array().inverse(), 0))
              .asDiagonal() *
          saes_mf.eigenvectors().transpose();
      bmm = b.segment(0, mf);
      Amr = A.block(0, mf, mf, n);
      Arm = A.block(mf, 0, n, mf);
      Arr = A.block(mf, mf, n, n);
      brr = b.segment(mf, n);
      A = (Arr - Arm * Amfmf_inv * Amr).eval();
      b = (brr - Arm * Amfmf_inv * bmm).eval();
    }
  }
  inverse_timer.Stop();
  DLOG(INFO) << "size: " << A.rows() << "x" << A.cols();
  DLOG(INFO) << "A: " << A.topLeftCorner(24, 24);

  // FIXME: IMPORTANT ERROR HERE!
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes2(A);
  Eigen::VectorXd S =
      Eigen::VectorXd((saes2.eigenvalues().array() > eps)
                          .select(saes2.eigenvalues().array(), 0));
  Eigen::VectorXd S_inv =
      Eigen::VectorXd((saes2.eigenvalues().array() > eps)
                          .select(saes2.eigenvalues().array().inverse(), 0));

  Eigen::VectorXd S_sqrt = S.cwiseSqrt();
  Eigen::VectorXd S_inv_sqrt = S_inv.cwiseSqrt();

  linearized_jacobians = S_sqrt.asDiagonal() * saes2.eigenvectors().transpose();
  linearized_residuals =
      S_inv_sqrt.asDiagonal() * saes2.eigenvectors().transpose() * b;
  DLOG(INFO) << "linearized_jacobians: " << linearized_jacobians.rows() << "x"
             << linearized_jacobians.cols();

  DLOG(INFO) << "linearized_residuals: " << linearized_residuals.rows() << "x"
             << linearized_residuals.cols();

  LOG(INFO) << "linearized_residuals: "
            << (linearized_residuals.transpose() * linearized_residuals);
  // LOG(INFO)
  //     << "ideal dx: " << std::fixed << std::setprecision(5)
  //     << (-(linearized_jacobians.transpose() *
  //     linearized_jacobians).inverse() *
  //         linearized_jacobians.transpose() * linearized_residuals)
  //            .transpose();
}

void MarginalizationInfo::ResetParameterBlockInfo() {
  // NOTE: factor and previous set & map release!!!

  for (int i = 0; i < factors.size(); i++) {
    delete[] factors[i]->raw_jacobians;

    factors[i].reset();
  }
  factors.clear();

  parameter_block_spec_set.clear();
  parameter_block_spec_pos_map.clear();
}

std::vector<ParameterBlockSpec *>
MarginalizationInfo::GetParameterBlockCollection() {
  keep_block_collection.clear();
  keep_block_spec_pos_map.clear();
  sum_block_size = 0;
  for (const auto &it : parameter_block_spec_pos_map) {
    if (it.second >= m) {
      // block info and data from old address
      // NOTE: it.first = old address, it.second = order in marginalization
      keep_block_collection.emplace_back(it.first);
      keep_block_spec_pos_map[it.first] = it.second;
      sum_block_size += it.first->GetMinimalDimension();
    }
  }

  return keep_block_collection;
}

/**
 * copy only the sizes of parameter block and residual block
 * @param _marginalization_info last info includes the # of r and x
 */
MarginalizationFactor::MarginalizationFactor(
    MarginalizationInfo *_marginalization_info)
    : marginalization_info(_marginalization_info) {
  int cnt = 0;
  std::stringstream ss, ss2;
  ss << "sizes: ";
  ss2 << "idx: ";
  int idx = 0;
  for (auto it : marginalization_info->keep_block_collection) {
    mutable_parameter_block_sizes()->push_back(it->GetDimension());
    cnt += it->GetDimension();
    ss << it << " ";
    ss2 << marginalization_info->keep_block_spec_pos_map[it] << " ";
  }
  DLOG(INFO) << ss.str();
  DLOG(INFO) << ss2.str();
  // printf("residual size: %d, %d\n", cnt, marginalization_info->n);
  set_num_residuals(marginalization_info->n);
}

bool MarginalizationFactor::Evaluate(double const *const *parameters,
                                     double *residuals,
                                     double **jacobians) const {
  int n = marginalization_info->n;
  int m = marginalization_info->m;
  Eigen::VectorXd dx(n);
  Eigen::VectorXd x_all(n);
  Eigen::VectorXd x0_all(n);
  std::stringstream ssdx, ssx, ssx0, ssab, ssabx0, ssabx;
  for (int i = 0; i < marginalization_info->keep_block_collection.size(); ++i) {
    ParameterBlockSpec *parameter_block_spec =
        marginalization_info->keep_block_collection[i];
    int size = parameter_block_spec->GetDimension();
    int idx =
        marginalization_info->keep_block_spec_pos_map[parameter_block_spec] - m;
    Eigen::VectorXd x = Eigen::Map<const Eigen::VectorXd>(parameters[i], size);
    Eigen::VectorXd x0 = Eigen::Map<const Eigen::VectorXd>(
        parameter_block_spec->GetParametersToZero(), size);
    // FIXME: replace with function minus in ParameterBlockSpec
    if (size == 2) {
      dx.segment(idx, size) = x - x0;
      x_all.segment(idx, size) = x;
      x0_all.segment(idx, size) = x0;
      if (!settigNoScalingAtOpt) {
        dx[idx] /= SCALE_A;
        dx[idx + 1] /= SCALE_B;
      }
      ssab << (x - x0).transpose()
           << (parameter_block_spec->IsFixed() ? " fixed" : " not fixed")
           << "; ";
      ssabx0 << x0.transpose() << "; ";
      ssabx << x.transpose() << "; ";
    } else if (size == 6) {
      // FIXME: use se3 to update
      // PRE_world_to_cam = SE3::exp(W2CLeftEps()) * GetWorldToCamEvalPT();
      dx.segment<6>(idx) = (SE3::exp(x) * SE3::exp(x0).inverse()).log();
      x_all.segment<6>(idx) = x;
      x0_all.segment<6>(idx) = x0;
    } else {
      dx.segment(idx, size) = x - x0;
      x_all.segment(idx, size) = x;
      x0_all.segment(idx, size) = x0;
    }
    ssdx << std::fixed << std::setprecision(5)
         << dx.segment(idx, size).transpose() << std::endl;
    ssx << std::fixed << std::setprecision(5) << x.transpose() << std::endl;
    ssx0 << std::fixed << std::setprecision(5) << x0.transpose() << std::endl;
  }

  if (!marginalization_info->evaluated) {
    marginalization_info->evaluated = true;
    DLOG(INFO) << "keep_block_collection.size(): "
               << marginalization_info->keep_block_collection.size();
    if (dx.norm() > 1e-5) {
      DLOG(ERROR) << dx.norm() << " ERROR";
    }
    DLOG(INFO) << "dx: " << ssdx.str();
    DLOG(INFO) << "x: " << ssx.str();
    DLOG(INFO) << "x0: " << ssx0.str();
  }
  LOG(INFO) << "ssab: " << ssab.str();
  // LOG(INFO) << "ssabx: " << ssabx.str();
  // LOG(INFO) << "ssabx0: " << ssabx0.str();

  Eigen::Map<Eigen::VectorXd>(residuals, n) =
      marginalization_info->linearized_residuals +
      marginalization_info->linearized_jacobians * dx;

  DLOG(INFO) << "linearized_residuals.norm(): "
             << marginalization_info->linearized_residuals.norm();
  DLOG(INFO) << "residuals.norm(): "
             << Eigen::Map<Eigen::VectorXd>(residuals, n).norm();

  DLOG(INFO) << "linearized_residuals: "
             << marginalization_info->linearized_residuals.norm();
  DLOG(INFO) << "linearized_residuals size: "
             << marginalization_info->linearized_residuals.rows();
  DLOG(INFO) << "dr: "
             << (marginalization_info->linearized_jacobians * dx).norm();
  if (jacobians) {
    for (int i = 0; i < marginalization_info->keep_block_collection.size();
         ++i) {
      ParameterBlockSpec *parameter_block_spec =
          marginalization_info->keep_block_collection[i];
      if (jacobians[i]) {
        int size = parameter_block_spec->GetDimension(),
            local_size = parameter_block_spec->GetMinimalDimension();
        int idx = marginalization_info
                      ->keep_block_spec_pos_map[parameter_block_spec] -
                  m;
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                 Eigen::RowMajor>>
            jacobian(jacobians[i], n, size);
        jacobian.setZero();
        jacobian.leftCols(local_size) =
            marginalization_info->linearized_jacobians.middleCols(idx,
                                                                  local_size);
      }
    }
  }
  return true;
}

}  // namespace dsl
