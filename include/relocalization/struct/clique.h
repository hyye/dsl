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

#ifndef DSL_CLIQUE_H
#define DSL_CLIQUE_H

#include "relocalization/relocalization_struct.h"
#include "relocalization/vocabulary_binary.h"
#include "relocalization/boost_archiver.h"

namespace dsl::relocalization {

class Clique {
 public:
  std::vector<unsigned long> mvNId;
  // Bag of Words Vector structures.
  DBoW2::BowVector mBowVec;
  DBoW2::FeatureVector mFeatVec;
  int mnRelocWords = 0;
  int mnRelocQuery = -1;
  float mRelocScore = 0;

  cv::Mat mMeanWorldPos;

  // ORB descriptor, each row associated to a keypoint.
  cv::Mat mDescriptors;
 private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive &ar, const unsigned int version);
};

class CliqueMap {
 public:
  CliqueMap(OrbVocabularyBinary *voc);
  std::vector<Clique> mvCliques;
  std::vector<std::unique_ptr<Clique>> mCliques;
  std::map<long, unsigned long> mVertIdToNId;
  std::map<unsigned long, long> mNIdToVertId;
  std::set<unsigned long> msNId;
  OrbVocabularyBinary *mpVoc;
  std::vector<std::list<Clique *> > mvInvertedFile;
  void ComputeCliques(std::unordered_map<unsigned long, std::unique_ptr<MapPoint>> &all_map_points,
                      int min_common_weight = 1);
  static void FindMaxCliqueIds(std::unordered_map<unsigned long, std::unique_ptr<MapPoint>> &all_map_points,
                               unsigned long mp_nid,
                               std::vector<unsigned long> &max_clique_nids,
                               int min_common_weight);
  void Save(std::string filename);
  bool Load(std::string filename);
  void BuildInvertedFile();
  std::vector<Clique *> DetectRelocalizationCandidates(Frame *pF);

 private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive &ar, const unsigned int version);
};

} // namespace dsl::relocalization


#endif // DSL_CLIQUE_H
