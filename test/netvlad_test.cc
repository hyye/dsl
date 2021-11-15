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
// Created by hyye on 7/23/20.
//

#include <gtest/gtest.h>
#include <glog/logging.h>
#include <boost/filesystem.hpp>

#include "fmt/format.h"
#include "relocalization/struct/net_vlad_utils.h"
#include "relocalization/desc_map.h"
#include "tool/euroc_reader.h"
#include "relocalization/relocalization_config.h"
#include "relocalization/visualization/drawer.h"
#include "relocalization/struct/vlad_database.h"
#include "util/timing.h"

namespace fs = boost::filesystem;

using namespace dsl;
using namespace dsl::relocalization;

// TEST(TEST_VLAD, TEST_LOAD) {
//   fs::path bin_folder("/mnt/HDD2/vlad_results/V102_left_pinhole_batch");
//   fs::path txt_folder("/mnt/HDD2/vlad_results/V102_left_pinhole_batch_txt");
//   for (const auto &entry : fs::directory_iterator(bin_folder)) {
//     std::string fn = entry.path().string();
//     cv::Mat vlad = NetVladUtils::ReadVLADBinary(fn);
//     std::ofstream ofs;
//     fn = fs::basename(fn);
//     fn.replace(19, 4, ".txt");
//     LOG(INFO) << fn;
//     ofs.open((txt_folder / fn).string());
//     LOG(INFO) << vlad.cols;
//     for (int i = 0; i < vlad.cols; ++i) {
//       ofs << vlad.at<float>(cv::Point(i, 0)) << " ";
//     }
//     ofs << std::endl;
//     ofs.close();
//   }
// }

TEST(TEST_VLAD, TEST_MATCHES) {
  // RelocalizationConfig
  //     yaml_loader
  //     ("/mnt/HDD/Datasets/Visual/EuRoC/V2_03_difficult/mav0/processed_data/relocalization/config.yaml");
  RelocalizationConfig
      yaml_loader
      ("/mnt/HDD/Datasets/Visual/EuRoC/V2_01_easy/mav0/processed_data/left_pinhole/relocalization/config.yaml");

  EurocReader reader(yaml_loader.query_dataset_path);
  EurocReader db_reader(yaml_loader.database_dataset_path);
  DescMap desc_map;
  desc_map.LoadMap(yaml_loader.database_map_path);
  fs::path vlad_db_path = yaml_loader.database_vlad_path;
  fs::path vlad_query_path = yaml_loader.query_vlad_path;

  LOG(INFO) << yaml_loader.database_vlad_path << " " << yaml_loader.query_vlad_path;

  desc_map.SetVLADPath(yaml_loader.database_vlad_path);

  VLADDatabase *vlad_database_ptr = desc_map.vlad_database_ptr.get();
  cv::Mat vlad_db = vlad_database_ptr->vlad_descs;

  LOG(INFO) << vlad_db.size();
  cv::flann::Index &flann_index = *vlad_database_ptr->flann_vlad_descs_ptr;
  cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);

  vlad_database_ptr->SetQueryDatabase(yaml_loader.query_vlad_path);
  ORBextractor feature_extractor(1000, 1.2, 8, 20, 7);
  OrbVocabularyBinary voc;
  voc.loadFromBinaryFile(yaml_loader.database_voc_path);

  for (int i = 0; i < reader.filenames.size(); ++i) {
    std::string fn = reader.filenames[i];

    std::string query_vlad_fn = (vlad_query_path / (fn + ".bin")).string();
    cv::Mat vlad_query = NetVladUtils::ReadVLADBinary(query_vlad_fn);
    reader.ReadImage(fn);

    Frame f(reader.gray_image, fn, &feature_extractor, &voc);

    timing::Timer timer("flann");
    std::vector<Frame *> result_frames = vlad_database_ptr->DetectRelocalizationCandidates(&f);
    timer.Stop();

    LOG(INFO) << timing::Timing::Print();

    // db_reader.ReadImage(knn_matches[0][0].trainIdx);
    // cv::namedWindow("matcher", cv::WINDOW_GUI_NORMAL);
    // cv::imshow("matcher", db_reader.gray_image);

    std::vector<cv::Mat> imgs;
    imgs.push_back(reader.gray_image.clone());

    LOG(INFO) << result_frames[0]->mTimeStamp;

    for (auto &&pKF : result_frames) {
      db_reader.ReadImage(pKF->mTimeStamp); // filenames[knn_matches[0][0].trainIdx]
      imgs.push_back(db_reader.gray_image.clone());
      break;
    }
    db_reader.ReadImage(vlad_database_ptr->result_knn.frames[0]->mTimeStamp);
    imgs.push_back(db_reader.gray_image.clone());
    cv::Mat draw = MakeCanvas(imgs, 300, 1);

    cv::namedWindow("q", cv::WINDOW_GUI_NORMAL);
    cv::imshow("q", draw);
    cv::waitKey(0);
  }
}
