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
// Created by hyye on 8/8/20.
//

#ifndef DSL_SP_MATCHER_H
#define DSL_SP_MATCHER_H

#include "relocalization/feature_matcher.h"

namespace dsl::relocalization {

class SPMatcher : public ORBmatcher {
 public:
  SPMatcher(float nnratio = 0.6);

  static float DescriptorDistance(const cv::Mat &a, const cv::Mat &b);
  int SearchByProjection(Frame &F, const std::vector<MapPoint *> &vpMapPoints, const float th = 3) override;
  // int SearchByProjection(Frame &F,
  //                        const std::vector<MapPoint *> &vpMapPoints,
  //                        std::vector<MapPoint *> &vpOutMPs,
  //                        const float th = 3);

  // Project MapPoints seen in KeyFrame into the Frame and search matches.
  // Used in relocalisation (Tracking)
  // int SearchByProjection(Frame &CurrentFrame, Frame *pKF, const std::set<MapPoint *> &sAlreadyFound,
  //                        const float th, const int ORBdist);

  int SearchByProjection(Frame &CurrentFrame,
                         const std::vector<MapPoint *> &vpMapPoints,
                         const std::set<MapPoint *> &sAlreadyFound,
                         const float th,
                         const float ORBdist) override;

  int SearchByBruteForce(const std::vector<MapPoint *> &vpMPs,
                         Frame &F,
                         std::vector<MapPoint *> &vpMapPointMatches,
                         bool useEnhancedMPs = true) override;
  int SearchByBruteForce(Frame *pKF, Frame &F, std::vector<MapPoint *> &vpMapPointMatches, bool useEnhancedMPs = true) override;
  // int SearchByBruteForce(Frame *pKF, Frame &F, std::vector<std::pair<int, int>> &corres_id, bool useEnhancedMPs);

  int Fuse(Frame *pF, const std::vector<MapPoint *> &vpMapPoints, const float th = 3.0) override;
  int FuseNew(Frame *pF, std::vector<MapPoint *> &vpMapPoints, const float th = 3.0) override;

 public:
  static const float TH_LOW;
  static const float TH_HIGH;
  static const int HISTO_LENGTH;
};

} // namespace

#endif // DSL_SP_MATCHER_H
