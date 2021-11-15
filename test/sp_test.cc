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
#include <gtest/gtest.h>
#include <boost/filesystem.hpp>

#include "fmt/format.h"
#include "relocalization/sp_extractor.h"
#include "relocalization/sp_matcher.h"
#include "tool/euroc_reader.h"
#include "relocalization/relocalization_config.h"
#include "relocalization/visualization/drawer.h"
#include "util/timing.h"
#include "util/global_calib.h"

#include <glog/logging.h>

namespace fs = boost::filesystem;

using namespace dsl;
using namespace dsl::relocalization;

TEST(TEST_SP, TEST_EXTRACTOR) {
  RelocalizationConfig
      yaml_loader
      ("/mnt/HDD/Datasets/Visual/EuRoC/V2_01_easy/mav0/processed_data/left_pinhole/relocalization/config.yaml");

  EurocReader reader(yaml_loader.query_dataset_path);

  Eigen::Matrix3f K;
  K << yaml_loader.gamma1, 0, yaml_loader.u0, 0, yaml_loader.gamma2,
       yaml_loader.v0, 0, 0, 1;
  SetGlobalCalib(yaml_loader.image_width, yaml_loader.image_height, K,
                 yaml_loader.xi);
  LOG(WARNING) << yaml_loader.Print();

  SPExtractor::PyTorchDevice pytorch_device = yaml_loader.pytorch_device == "CUDA" ? SPExtractor::PyTorchDevice::CUDA : SPExtractor::PyTorchDevice::CPU;
  // NOTE: the feature number 1000 is unused
  SPExtractor sp_extractor(1000, yaml_loader.superpoint_path, pytorch_device);

  LOG(INFO) << "Testing...";

  for (int i = 0; i < reader.filenames.size(); ++i) {
    std::string fn = reader.filenames[i];

    reader.ReadImage(fn);

    timing::Timer timer("sp");
    Frame f(reader.gray_image, fn, &sp_extractor, NULL);
    std::vector<cv::KeyPoint> vec_kpt = f.mvKeysUn;
    cv::Mat desc = f.mDescriptors;

    cv::Mat desc1 = desc.row(0);
    cv::Mat desc2 = desc.row(1);
    LOG(WARNING) << "dist: " << SPMatcher::DescriptorDistance(desc1, desc2);
    LOG(WARNING) << vec_kpt[0].octave << " " << vec_kpt[0].size;
    LOG(WARNING) << vec_kpt[1].octave << " " << vec_kpt[1].size;

    // sp_extractor(reader.gray_image, cv::Mat(), vec_kpt, desc);
    timer.Stop();

    cv::Mat img_plot;
    cv::drawKeypoints(reader.gray_image, vec_kpt, img_plot,
                      cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    LOG(WARNING) << vec_kpt.size();

    LOG(WARNING) << timing::Timing::Print();
    
    cv::namedWindow("q", cv::WINDOW_GUI_NORMAL);
    cv::imshow("q", img_plot);
    cv::waitKey(10);
  }
}

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_alsologtostderr = true;

  testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
