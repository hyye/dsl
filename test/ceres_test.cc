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
// Created by hyye on 11/17/19.
//

#include <ceres/ceres.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <sophus/se3.hpp>
#include "full_system/full_system.h"
#include "optimization/photo_cost_functor.h"
#include "optimization/se3_local_parameterization.h"
#include "optimization/se3_prior_cost_functor.h"
#include "util/timing.h"

#include "tool/new_tsukuba_reader.h"

using namespace dsl;
using std::cout;
using std::endl;

typedef Eigen::Matrix<double, 6, 1> Vec6;
typedef Eigen::Matrix<double, 3, 1> Vec3;
typedef Sophus::SE3d SE3d;

struct CostFunctor {
  template <typename T>
  bool operator()(const T* const x, T* residual) const {
    residual[0] = T(10.0) - x[0];
    return true;
  }
};

struct SE3CostFunctor : public ceres::SizedCostFunction<3, 6> {
  SE3CostFunctor(const Vec3& p, const Vec3& o) : point(p), observed(o) {}
  Vec3 point, observed;

  template <typename T>
  bool operator()(const T* const x, T* residual) const {
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals(residual);
    Eigen::Map<const Eigen::Matrix<T, 6, 1>> se3_log(x);
    Sophus::SE3<T> transform = Sophus::SE3<T>::exp(se3_log);
    residuals = observed.cast<T>() - transform * point.cast<T>();
    return true;
  }

  virtual ~SE3CostFunctor() {}
  virtual bool Evaluate(double const* const* parameters, double* residuals,
                        double** jacobians) const {
    Eigen::Map<Eigen::Matrix<double, 3, 1>> r(residuals);
    Eigen::Map<const Eigen::Matrix<double, 6, 1>> se3_log(parameters[0]);
    SE3d transform = SE3d::exp(se3_log);
    r = observed - transform * point;

    // Compute the Jacobian if asked for.
    if (jacobians != NULL && jacobians[0] != NULL) {
      Eigen::Map<Eigen::Matrix<double, 3, 6, Eigen::RowMajor>> j(jacobians[0]);
      j.leftCols<3>() = -Eigen::Matrix3d::Identity();
      j.rightCols<3>() = Sophus::SO3d::hat(transform * point);
    }
    return true;
  }
};

struct TwoBlocksCostFunctor : public ceres::SizedCostFunction<3, 6, 6> {
  TwoBlocksCostFunctor(const Vec3& p, const Vec3& o) : point(p), observed(o) {}
  Vec3 point, observed;

  virtual ~TwoBlocksCostFunctor() {}
  virtual bool Evaluate(double const* const* parameters, double* residuals,
                        double** jacobians) const {
    Eigen::Map<Eigen::Matrix<double, 3, 1>> r(residuals);
    Eigen::Map<const Eigen::Matrix<double, 6, 1>> se3_0(parameters[0]);
    Eigen::Map<const Eigen::Matrix<double, 6, 1>> se3_1(parameters[1]);
    SE3d transform0 = SE3d::exp(se3_0);
    SE3d transform1 = SE3d::exp(se3_1);
    r = observed - transform1 * transform0 * point;

    // Compute the Jacobian if asked for.
    if (jacobians != NULL) {
      if (jacobians[0] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 3, 6, Eigen::RowMajor>> j(
            jacobians[0]);
        Eigen::Matrix3d R1 = transform1.rotationMatrix();
        j.leftCols<3>() = -R1;
        j.rightCols<3>() = R1 * Sophus::SO3d::hat(transform0 * point);
      }
      if (jacobians[1] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 3, 6, Eigen::RowMajor>> j(
            jacobians[1]);
        j.leftCols<3>() = -Eigen::Matrix3d::Identity();
        j.rightCols<3>() = Sophus::SO3d::hat(transform1 * transform0 * point);
      }
    }
    return true;
  }
};

struct MyCallBack : public ceres::IterationCallback {
 public:
  MyCallBack(Vec6& tan0, Vec6& tan1) : tan0_(tan0), tan1_(tan1) {}
  virtual ceres::CallbackReturnType operator()(
      const ceres::IterationSummary& summary) {
    LOG(INFO) << "MyCallBack";
    LOG(INFO) << (SE3d::exp(tan1_) * SE3d::exp(tan0_)).log().transpose();
    return ceres::SOLVER_CONTINUE;
  }
  Vec6 &tan0_, &tan1_;
};

static std::vector<Vec3> points, observations, obs_points;
static SE3d T_gt0, T_gt1;
static SE3d T_noise0, T_noise1;
const int N = 100;

TEST(CeresTest, BasicTest) {
  // The variable to solve for with its initial value.
  double initial_x = 5.0;
  double x = initial_x;

  // Build the problem.
  ceres::Problem problem;

  // Set up the only cost function (also known as residual). This uses
  // auto-differentiation to obtain the derivative (jacobian).
  ceres::CostFunction* cost_function =
      new ceres::AutoDiffCostFunction<CostFunctor, 1, 1>(new CostFunctor);
  problem.AddResidualBlock(cost_function, new ceres::CauchyLoss(0.5), &x);

  // Run the solver!
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;
  Solve(options, &problem, &summary);

  LOG(INFO) << summary.BriefReport() << "\n";
  LOG(INFO) << "x : " << initial_x << " -> " << x << "\n";
}

TEST(CeresTest, EigenTest) {
  double array[7][6];
  std::array<std::array<double, 6>, 7> se3_array{};
  int idx = 0;
  for (auto&& se3 : se3_array) {
    for (int i = 0; i < 6; ++i) {
      se3[i] = i;
      array[idx][i] = i;
    }
    ++idx;
  }
  Eigen::Matrix<double, 7, 6> v =
      Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>>(
          se3_array.data()->data());
  LOG(INFO) << v << endl;
  LOG(INFO) << "===" << endl;
  LOG(INFO) << Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>>(
                   &(**array))
            << endl;
}

TEST(CeresTest, AutoTest) {
  Vec6 T_tangent = T_noise0.log();
  ceres::Problem problem;

  for (int i = 0; i < N; ++i) {
    ceres::CostFunction* cost_function =
        new ceres::AutoDiffCostFunction<SE3CostFunctor, 3, 6>(
            new SE3CostFunctor(points[i], observations[i]));
    problem.AddResidualBlock(cost_function, NULL, T_tangent.data());
  }

  LOG(INFO) << "initial" << endl << T_tangent.transpose();

  // Run the solver!
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = false;
  ceres::Solver::Summary summary;

  timing::Timer timer("automatic");
  Solve(options, &problem, &summary);
  timer.Stop();

  LOG(INFO) << summary.BriefReport() << "\n";
  LOG(INFO) << "optimized" << endl << T_tangent.transpose();
  LOG(INFO) << "gt" << endl << T_gt0.log().transpose();
  LOG(INFO) << timing::Timing::GetTotalSeconds("automatic");
}

TEST(CeresTest, AnalyticalTest) {
  Vec6 T_tangent = T_noise0.log();

  ceres::Problem problem;

  ceres::LocalParameterization* local_parameterization =
      new SE3LocalParameterization();
  problem.AddParameterBlock(T_tangent.data(), 6, local_parameterization);
  for (int i = 0; i < N; ++i) {
    ceres::CostFunction* cost_function =
        new SE3CostFunctor(points[i], observations[i]);
    problem.AddResidualBlock(cost_function, NULL, T_tangent.data());
  }

  LOG(INFO) << "initial" << endl << T_tangent.transpose();

  // Run the solver!
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = false;
  ceres::Solver::Summary summary;
  timing::Timer timer("analytical");
  Solve(options, &problem, &summary);
  timer.Stop();

  LOG(INFO) << summary.BriefReport() << "\n";
  LOG(INFO) << "optimized" << endl << T_tangent.transpose();
  LOG(INFO) << "gt" << endl << T_gt0.log().transpose();
  LOG(INFO) << timing::Timing::GetTotalSeconds("analytical");

  LOG(INFO) << timing::Timing::Print();
}

TEST(CeresTest, TwoParamBlockTest) {
  Vec6 T_tangent0 = T_noise0.log();
  Vec6 T_tangent1 = T_noise1.log();

  ceres::Problem problem;

  ceres::LocalParameterization* local_parameterization =
      new SE3LocalParameterization();
  problem.AddParameterBlock(T_tangent0.data(), 6, local_parameterization);
  problem.AddParameterBlock(T_tangent1.data(), 6, local_parameterization);
  for (int i = 0; i < N; ++i) {
    ceres::CostFunction* cost_function =
        new TwoBlocksCostFunctor(points[i], obs_points[i]);
    problem.AddResidualBlock(cost_function, NULL, T_tangent0.data(),
                             T_tangent1.data());
  }

  LOG(INFO)
      << "before:"
      << (SE3d::exp(T_tangent1) * SE3d::exp(T_tangent0)).log().transpose();

  // Run the solver!
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = false;
  options.update_state_every_iteration = true;
  ceres::IterationCallback* cb = new MyCallBack(T_tangent0, T_tangent1);
  options.callbacks.push_back(cb);
  ceres::Solver::Summary summary;

  Solve(options, &problem, &summary);

  LOG(INFO) << summary.BriefReport() << "\n";

  LOG(INFO)
      << "optimized:"
      << (SE3d::exp(T_tangent1) * SE3d::exp(T_tangent0)).log().transpose();
  LOG(INFO) << "multi gt:" << (T_gt1 * T_gt0).log().transpose();
}

TEST(CeresTest, SE3PriorCostFunctorTest) {
  Eigen::Matrix<double, 6, 6> information =
      Eigen::Matrix<double, 6, 6>::Identity() * 1e8;
  SE3 measurement = SE3d::exp(Vec6::Random());
  SE3 n_measurement = SE3d::exp(Vec6::Random() * 0.05) * measurement;
  SE3PriorCostFunctor functor(measurement, information);
  Eigen::Matrix<double, 6, 1> n_measurement_vec = n_measurement.log();
  Eigen::Matrix<double, 6, 1> residual;
  Eigen::Matrix<double, 6, 6, Eigen::RowMajor> J0;
  std::vector<double*> parameters{n_measurement_vec.data()};
  std::vector<double*> jacobians{J0.data()};

  functor.Evaluate(parameters.data(), residual.data(), jacobians.data());
  LOG(INFO) << "analytic res: " << std::endl
            << residual.transpose() << std::endl;
  LOG(INFO) << "analytic J: " << std::endl << J0 << std::endl;

  auto Plus = [](const SE3& x, const Eigen::Matrix<double, 6, 1>& dx,
                 SE3& result) { result = SE3::exp(dx) * x; };

  Eigen::LLT<Eigen::Matrix<double, 6, 6>> llt_of_information(information);
  Eigen::Matrix<double, 6, 6> sqrt_information =
      llt_of_information.matrixL().transpose();

  Eigen::Matrix<double, 6, 6, Eigen::RowMajor> Jpm;
  Eigen::Matrix<double, 6, 1> rn =
      sqrt_information * (measurement.inverse() * n_measurement).log();
  double dx = 1e-8;
  for (size_t i = 0; i < 6; ++i) {
    Eigen::Matrix<double, 6, 1> delta;
    SE3 xp, xm;
    delta.setZero();
    delta[i] = dx;
    Plus(n_measurement, delta, xp);
    Eigen::Matrix<double, 6, 1> Jp =
        sqrt_information * (measurement.inverse() * xp).log();
    delta[i] = -dx;
    Plus(n_measurement, delta, xm);
    Eigen::Matrix<double, 6, 1> Jm =
        sqrt_information * (measurement.inverse() * xm).log();
    Jpm.col(i) = (Jp - Jm) / (2 * dx);
  }

  LOG(INFO) << "analytic res: " << std::endl << rn.transpose() << std::endl;
  LOG(INFO) << "analytic J: " << std::endl << Jpm << std::endl;
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_alsologtostderr = true;

  testing::InitGoogleTest(&argc, argv);

  srand((unsigned int)time(0));

  T_gt0 = SE3d::exp(Vec6::Random());
  T_noise0 = SE3d::exp(Vec6::Random() * 0.05) * T_gt0;

  T_gt1 = SE3d::exp(Vec6::Random());
  T_noise1 = SE3d::exp(Vec6::Random() * 0.05) * T_gt1;

  for (int i = 0; i < N; ++i) {
    Vec3 p = Vec3::Random();
    points.emplace_back(p);
    observations.emplace_back(T_gt0 * p);
    obs_points.emplace_back(T_gt1 * T_gt0 * p);
  }

  return RUN_ALL_TESTS();
}