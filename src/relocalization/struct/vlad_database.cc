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

#include "relocalization/struct/vlad_database.h"
#include "dsl_common.h"
#include "fmt/format.h"

namespace fs = boost::filesystem;

namespace dsl::relocalization {

VLADBinaryLoader::VLADBinaryLoader(const std::string &path) : path_(path) {}

cv::Mat VLADBinaryLoader::Load(const std::string &timestamp) {
  std::string query_vlad_fn = (path_ / (timestamp + ".bin")).string();
  cv::Mat vlad_query = NetVladUtils::ReadVLADBinary(query_vlad_fn);
  return vlad_query;
}

VLADDatabase::VLADDatabase(const std::string &vlad_path, const std::vector<Frame *> &frames) :
    frames_(frames), vlad_binary_loader(vlad_path) {
  for (auto &&pF :frames) {
    std::string fn = pF->mTimeStamp;
    cv::Mat vlad = vlad_binary_loader.Load(fn);
    vlad_descs.push_back(vlad);
  }
  flann_vlad_descs_ptr = std::make_unique<cv::flann::Index>(vlad_descs, cv::flann::KDTreeIndexParams(1));
}

void VLADDatabase::SetQueryDatabase(const std::string &query_vlad_path) {
  query_vlad_binary_loader = VLADBinaryLoader(query_vlad_path);
}

VLADKnnResult VLADDatabase::QueryKnn(const cv::Mat &vlad_query, int knn) {
  std::vector<int> indices;
  std::vector<float> dists;
  flann_vlad_descs_ptr->knnSearch(vlad_query, indices, dists, knn);
  std::vector<Frame *> result_frames;
  for (int idx : indices) {
    result_frames.emplace_back(frames_[idx]);
  }
  std::vector<double> si;
  std::string dists_str;
  for (float d : dists) {
    double score = 2 - sqrt(d);
    dists_str += fmt::format("{} ", 2 - sqrt(d));
    si.push_back(score);
  }
  cv::Mat best_desc = vlad_descs.row(indices[0]);
  // LOG(INFO) << 2 - cv::norm(best_desc - vlad_query);
  // LOG(INFO) << dists_str;
  VLADKnnResult result;
  result.frames = result_frames;
  result.si = si;
  return result;
}

std::vector<Frame *> VLADDatabase::DetectRelocalizationCandidates(Frame *pF) {
  LOG_ASSERT(!query_vlad_binary_loader.path_.empty());
  cv::Mat vlad_query = query_vlad_binary_loader.Load(pF->mTimeStamp);

  result_knn = QueryKnn(vlad_query, 30);
  float max_si = *std::max_element(result_knn.si.begin(), result_knn.si.end());
  float min_si = max_si * 0.8f;
  LOG(INFO) << "max si: " << max_si;
  std::list<std::pair<float, Frame *>> lScoreAndMatch;
  for (int i = 0; i < result_knn.si.size(); ++i) {
    float s = result_knn.si[i];
    Frame *pKF = result_knn.frames[i];
    if (s <= min_si) continue;
    pKF->mRelocScore = s;
    pKF->mnRelocQuery = pF->mnId;
    lScoreAndMatch.emplace_back(s, pKF);
  }

  if (lScoreAndMatch.empty())
    return std::vector<Frame *>();

  std::set<Frame *> sBestFrames;
  std::vector<Frame *> vBestFrames;
  for (std::list<std::pair<float, Frame *> >::iterator it = lScoreAndMatch.begin(),
           itend = lScoreAndMatch.end();
       it != itend; it++) {
    Frame *pKFi = it->second;
    std::vector<Frame *> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);
    bool isInBest = false;
    for (auto &&pNKF:vpNeighs) {
      if (sBestFrames.count(pNKF)) {
        isInBest = true;
        break;
      }
    }
    if (!isInBest) {
      LOG_ASSERT(!sBestFrames.count(pKFi));
      sBestFrames.insert(pKFi);
      vBestFrames.push_back(pKFi);
    }
  }

  std::vector<Frame *>
      vpRelocCandidates(vBestFrames.begin(), vBestFrames.begin() + std::min(3, (int) vBestFrames.size()));

  LOG(INFO) << sBestFrames.size() << " " << vpRelocCandidates.size() << " " << pF->mnId;

  return vpRelocCandidates;
}

} // namespace dsl::relocalization