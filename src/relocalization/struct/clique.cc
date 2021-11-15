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
// Created by hyye on 7/10/20.
//

#include "relocalization/struct/clique.h"
#include <boost/serialization/unique_ptr.hpp>
#include "relocalization/converter.h"

namespace dsl::relocalization {

template<class Archive>
void Clique::serialize(Archive &ar, const unsigned int version) {
  ar & mvNId;
  ar & mBowVec;
  ar & mFeatVec;
  ar & mDescriptors;
  ar & mMeanWorldPos;
}

CliqueMap::CliqueMap(OrbVocabularyBinary *voc) : mpVoc(voc) {
  mvInvertedFile.resize(mpVoc->size());
}

void CliqueMap::ComputeCliques(
    std::unordered_map<unsigned long, std::unique_ptr<MapPoint>>
    &all_map_points,
    int min_common_weight) {
  // For random selection
  long vertId = 0;
  for (auto &&map_point : all_map_points) {
    MapPoint *pMP = map_point.second.get();
    LOG_ASSERT(map_point.first == pMP->GetId());
    mVertIdToNId.insert(
        std::pair<long, unsigned long>(vertId, map_point.first));
    mNIdToVertId.insert(
        std::pair<unsigned long, long>(map_point.first, vertId));
    vertId += 1;
  }
  int iter = 0;
  int total_cliques = 0;

  while (msNId.size() < all_map_points.size() * 0.5 &&
      iter < all_map_points.size() * 0.2) {
    LOG(INFO) << "iter: " << iter << " sNmId.size(): " << msNId.size()
              << " total_cliques: " << total_cliques;
    int rand_int = RandomInt(0, mVertIdToNId.size() - 1);
    // rand_int = iter;
    iter += 1;
    LOG_ASSERT(mVertIdToNId.count(rand_int));
    unsigned long mpid = mVertIdToNId[rand_int];
    LOG_ASSERT(all_map_points.count(mpid));

    if (msNId.count(rand_int)) continue;
    std::vector<unsigned long> max_clique_nids;
    FindMaxCliqueIds(all_map_points, mpid, max_clique_nids, min_common_weight);
    int size_clique = max_clique_nids.size();
    if (size_clique > 1) {
      int overlap = 0;
      for (auto &&it : max_clique_nids) {
        if (msNId.count(it)) overlap += 1;
      }
      if (double(overlap) / size_clique < 0.5) {
        for (auto &&it : max_clique_nids) msNId.insert(it);
        total_cliques += 1;
        std::unique_ptr<Clique> pClique = std::make_unique<Clique>();
        pClique->mvNId = max_clique_nids;
        cv::Mat meanWorldPos = (cv::Mat_<float>(3, 1) << 0, 0, 0);
        for (auto &&nid : max_clique_nids) {
          pClique->mDescriptors.push_back(all_map_points[nid]->GetDescriptor());
          meanWorldPos += all_map_points[nid]->GetWorldPos();
        }
        pClique->mMeanWorldPos = meanWorldPos / max_clique_nids.size();
        std::vector<cv::Mat> vCurrentDesc =
            Converter::toDescriptorVector(pClique->mDescriptors);
        mpVoc->transform(vCurrentDesc, pClique->mBowVec, pClique->mFeatVec, 4);
        mCliques.push_back(std::move(pClique));
      }
    }
  }
  BuildInvertedFile();
}

void CliqueMap::FindMaxCliqueIds(
    std::unordered_map<unsigned long, std::unique_ptr<MapPoint>>
    &all_map_points,
    unsigned long mp_nid, std::vector<unsigned long> &max_clique_nids,
    int min_common_weight) {
  LOG_ASSERT(all_map_points.count(mp_nid));
  max_clique_nids.clear();
  max_clique_nids.push_back(mp_nid);
  MapPoint *pMP = all_map_points[mp_nid].get();
  std::map<unsigned long, int> remaining_mpws = pMP->mConnectedMapPointWeights;
  auto comp = [](const std::pair<unsigned long, int> &p1,
                 const std::pair<unsigned long, int> &p2) {
    return p1.second < p2.second;
  };

  while (!remaining_mpws.empty()) {
    auto max_it =
        std::max_element(remaining_mpws.begin(), remaining_mpws.end(), comp);
    if (max_it != remaining_mpws.end()) {
      std::pair<unsigned long, int> best_mpw = *max_it;
      LOG_ASSERT(all_map_points.count(best_mpw.first));
      MapPoint *pBestMP = all_map_points[best_mpw.first].get();
      remaining_mpws.erase(max_it);
      bool valid = true;
      for (auto &&other_mpId : max_clique_nids) {
        if (!pBestMP->mConnectedMapPointWeights.count(other_mpId)) {
          valid = false;
        } else {
          int weight = pBestMP->mConnectedMapPointWeights[other_mpId];
          if (weight < min_common_weight) {
            valid = false;
          }
        }
      }
      if (valid) {
        max_clique_nids.push_back(best_mpw.first);
      }
    }
  }
}

std::vector<Clique *> CliqueMap::DetectRelocalizationCandidates(Frame *pF) {
  std::list<Clique *> lKFsSharingWords;

  // Search all cliques that share a word with the query frame
  for (DBoW2::BowVector::const_iterator vit = pF->mBowVec.begin(),
           vend = pF->mBowVec.end(); vit != vend; vit++) {
    std::list<Clique *> &lKFs = mvInvertedFile[vit->first];
    for (std::list<Clique *>::iterator lit = lKFs.begin(), lend = lKFs.end(); lit != lend; lit++) {
      Clique *pKFi = *lit;
      if (pKFi->mnRelocQuery != pF->mnId) {
        pKFi->mnRelocWords = 0;
        pKFi->mnRelocQuery = pF->mnId;
        lKFsSharingWords.push_back(pKFi);
      }
      ++pKFi->mnRelocWords;
    }
  }

  if (lKFsSharingWords.empty())
    return std::vector<Clique *>();

  // Adapted from ORB-SLAM2
  // Only compare against those keyframes that share enough words
  int maxCommonWords = 0;
  for (std::list<Clique *>::iterator lit = lKFsSharingWords.begin(), lend = lKFsSharingWords.end(); lit != lend;
       lit++) {
    if ((*lit)->mnRelocWords > maxCommonWords)
      maxCommonWords = (*lit)->mnRelocWords;
  }

  int minCommonWords = maxCommonWords * 0.8f;

  std::list<std::pair<float, Clique *> > lScoreAndMatch;
  float bestScore = 0;

  // Compute similarity score.
  for (std::list<Clique *>::iterator lit = lKFsSharingWords.begin(), lend = lKFsSharingWords.end(); lit != lend;
       lit++) {
    Clique *pKFi = *lit;

    if (pKFi->mnRelocWords > minCommonWords) {
      float si = mpVoc->score(pF->mBowVec, pKFi->mBowVec);
      pKFi->mRelocScore = si;
      lScoreAndMatch.push_back(std::make_pair(si, pKFi));
      if (si > bestScore) bestScore = si;
      LOG(INFO) << "si: " << si << " " << pKFi->mnRelocWords;
    }
  }

  // WARNING: no covisibility graph info here
  float minScoreToRetain = 0.75f * bestScore;
  std::set<Clique *> spAlreadyAddedKF;
  std::vector<Clique *> vpRelocCandidates;
  for (std::list<std::pair<float, Clique *> >::iterator it = lScoreAndMatch.begin(), itend = lScoreAndMatch.end(); it
      != itend;
       it++) {
    const float &si = it->first;
    if (si > minScoreToRetain) {
      Clique *pKFi = it->second;
      if (!spAlreadyAddedKF.count(pKFi)) {
        vpRelocCandidates.push_back(pKFi);
        spAlreadyAddedKF.insert(pKFi);
      }
    }
  }
  return vpRelocCandidates;
}

void CliqueMap::BuildInvertedFile() {
  for (auto &&c : mCliques) {
    for (DBoW2::BowVector::const_iterator vit = c->mBowVec.begin(),
             vend = c->mBowVec.end();
         vit != vend; vit++)
      mvInvertedFile[vit->first].push_back(c.get());
  }
}

void CliqueMap::Save(std::string filename) {
  std::ofstream out(filename, std::ios_base::binary);
  if (!out) {
    LOG(ERROR) << "Cannot Write to CliqueMap file: " << filename;
  }
  LOG(INFO) << "Saving CliqueMap: " << filename << std::flush;
  boost::archive::binary_oarchive oa(out, boost::archive::no_header);

  mvCliques.clear();
  for (auto &&c : mCliques) {
    mvCliques.push_back(*c);
  }

  oa << mvCliques;
  oa << mNIdToVertId;
  oa << mVertIdToNId;
  oa << msNId;

  LOG(INFO) << msNId.size() << std::endl;
  LOG(INFO) << mvCliques.size() << " ...done" << std::endl;
  out.close();
}

bool CliqueMap::Load(std::string filename) {
  std::ifstream in(filename, std::ios_base::binary);
  if (!in) {
    LOG(ERROR) << "Cannot Open CliqueMap file: " << filename
               << " , Create a new one";
    return false;
  }
  LOG(INFO) << "Loading CliqueMap: " << filename;
  boost::archive::binary_iarchive ia(in, boost::archive::no_header);
  mCliques.clear();
  mvCliques.clear();
  mNIdToVertId.clear();
  mVertIdToNId.clear();
  msNId.clear();

  ia >> mvCliques;
  ia >> mNIdToVertId;
  ia >> mVertIdToNId;
  ia >> msNId;

  for (auto &&c : mvCliques) {
    mCliques.emplace_back(std::make_unique<Clique>(c));
  }

  BuildInvertedFile();

  LOG(INFO) << msNId.size() << std::endl;
  LOG(INFO) << mCliques.size() << " ...done";

  in.close();
  return true;
}

template<class Archive>
void CliqueMap::serialize(Archive &ar, const unsigned int version) {
  ar & mCliques;
  ar & mNIdToVertId;
  ar & mVertIdToNId;
  ar & msNId;
}

template void CliqueMap::serialize(boost::archive::binary_iarchive &,
                                   const unsigned int);
template void CliqueMap::serialize(boost::archive::binary_oarchive &,
                                   const unsigned int);

}  // namespace dsl::relocalization