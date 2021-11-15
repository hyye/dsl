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
// Created by hyye on 11/6/19.
//

#include "util/index_thread_reduce.h"
#include <gtest/gtest.h>

using namespace dsl;

DEFINE_int64(N, 20, "input number");

static std::vector<int> global_v;

void SumReductor(int min, int max, int &stats, int tid) {
  for (int i = min; i < max; ++i) {
    stats += global_v[i];
  }
}

TEST(MTTest,IndexThreadReduceTest) {
  int N = FLAGS_N;
  for (int i = 0; i < N; ++i) {
    global_v.emplace_back(i * 1);
  }

  IndexThreadReduce<int> thread_reduce;
  thread_reduce.Reduce(
      std::bind(&SumReductor, std::placeholders::_1, std::placeholders::_2,
                std::placeholders::_3, std::placeholders::_4),
      0, 1, 0);
  int stats = 0;
  if (thread_reduce.GetStats(stats)) {
    // std::cout << "Successful" << std::endl;
  } else {
    // std::cout << "Unsuccessful" << std::endl;
  }
  LOG(INFO) << "stats: " << stats;

  thread_reduce.Reduce(
      std::bind(&SumReductor, std::placeholders::_1, std::placeholders::_2,
                std::placeholders::_3, std::placeholders::_4),
      1, N, 3);

  stats = 0;
  if (thread_reduce.GetStats(stats)) {
    // std::cout << "Successful" << std::endl;
  } else {
    // std::cout << "Unsuccessful" << std::endl;
  }
  LOG(INFO) << "stats: " << stats;
}

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_alsologtostderr = true;

  testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
