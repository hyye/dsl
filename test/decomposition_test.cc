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
// Created by hyye on 11/26/19.
//

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <Eigen/Eigen>

using std::endl;

TEST(DecompositionTest, ResJacobTest) {
  double eps = 1e-8;
  Eigen::Matrix<double, 3, 7> J1;
  Eigen::Matrix<double, 3, 7> J2;
  Eigen::Matrix<double, 3, 7> J3;
  Eigen::Matrix<double, 7, 7> J_star;
  Eigen::Matrix<double, 3, 1> res1;
  Eigen::Matrix<double, 3, 1> res2;
  Eigen::Matrix<double, 3, 1> res3;
  Eigen::Matrix<double, 7, 1> res_star;

  J1.setRandom();
  res1.setRandom();
  J2.setRandom();
  res2.setRandom();
  J3.setRandom();
  res3.setRandom();
  Eigen::MatrixXd A =
      J1.transpose() * J1 + J2.transpose() * J2 + J3.transpose() * J3;
  Eigen::MatrixXd b =
      J1.transpose() * res1 + J2.transpose() * res2 + J3.transpose() * res3;
  double rTr = res1.dot(res1) + res2.dot(res2) + res3.dot(res3);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes2(A);
  Eigen::VectorXd S =
      Eigen::VectorXd((saes2.eigenvalues().array() > eps)
                          .select(saes2.eigenvalues().array(), 0));
  Eigen::VectorXd S_inv =
      Eigen::VectorXd((saes2.eigenvalues().array() > eps)
                          .select(saes2.eigenvalues().array().inverse(), 0));

  Eigen::VectorXd S_sqrt = S.cwiseSqrt();
  Eigen::VectorXd S_inv_sqrt = S_inv.cwiseSqrt();

  J_star = S_sqrt.asDiagonal() * saes2.eigenvectors().transpose();
  res_star = S_inv_sqrt.asDiagonal() * saes2.eigenvectors().transpose() * b;
  LOG(INFO) << " S_sqrt.asDiagonal()" << endl
            << Eigen::MatrixXd(S_sqrt.asDiagonal());
  LOG(INFO) << "saes2.eigenvectors().transpose()" << endl
            << saes2.eigenvectors().transpose();
  LOG(INFO) << "J_star" << endl << J_star;
  LOG(INFO) << "res_star" << endl << res_star;

  LOG(INFO) << "===";

  LOG(INFO) << "JTJ - JsTJs:" << endl
            << (A - J_star.transpose() * J_star).sum();

  LOG(INFO) << "JTr - JsTrs:" << endl
            << (b - J_star.transpose() * res_star).sum();

  LOG(INFO) << "rTr:" << endl
            << rTr << endl
            << "rsTrs:" << endl
            << res_star.transpose() * res_star;
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_alsologtostderr = true;

  testing::InitGoogleTest(&argc, argv);
  srand((unsigned int)time(0));

  return RUN_ALL_TESTS();
}