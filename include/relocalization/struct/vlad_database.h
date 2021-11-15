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
// Created by hyye on 7/24/20.
//

#ifndef DSL_VLAD_DATABASE_H
#define DSL_VLAD_DATABASE_H

#include "boost/filesystem.hpp"
#include "relocalization/struct/net_vlad_utils.h"
#include "relocalization/struct/frame.h"

namespace dsl::relocalization {

class VLADBinaryLoader{
 public:
  VLADBinaryLoader() {};
  VLADBinaryLoader(const std::string & path);
  cv::Mat Load(const std::string& timestamp);

  boost::filesystem::path path_;
};

struct VLADKnnResult {
  std::vector<Frame *> frames;
  std::vector<double> si;
};

class VLADDatabase {
 public:
  VLADDatabase(const std::string &vlad_path, const std::vector<Frame *> &frames);
  void SetQueryDatabase(const std::string &query_vlad_path);
  VLADKnnResult QueryKnn(const cv::Mat &vlad_query, int knn = 20);
  std::vector<Frame *> DetectRelocalizationCandidates(Frame *pF);
  cv::Mat vlad_descs;
  std::unique_ptr<cv::flann::Index> flann_vlad_descs_ptr;
  VLADBinaryLoader vlad_binary_loader;
  VLADBinaryLoader query_vlad_binary_loader;

  VLADKnnResult result_knn;

  std::vector<Frame *> frames_;
};

} // namespace dsl::relocalization

#endif // DSL_VLAD_DATABASE_H
