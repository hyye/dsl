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
// Created by hyye on 7/1/20.
//

#ifndef DSL_FEATURE_MATCHER_H
#define DSL_FEATURE_MATCHER_H

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "relocalization/relocalization_struct.h"
#include "relocalization/desc_map.h"

namespace dsl {

namespace relocalization {

class FeatureMatcher {};

// Adapted from ORB-SLAM2
class ORBmatcher : public FeatureMatcher {
 public:

  ORBmatcher(float nnratio = 0.6, bool checkOri = true);

  // Computes the Hamming distance between two ORB descriptors
  static int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);

  bool SearchFromFeatVec(Frame &F, std::vector<unsigned int> &feat_ids, cv::Mat &query_descriptor, int &feat_id);

  // Search matches between Frame keypoints and projected MapPoints. Returns number of matches
  // Used to track the local map (Tracking)
  virtual int SearchByProjection(Frame &F, const std::vector<MapPoint *> &vpMapPoints, const float th = 3);
  int SearchByProjection(Frame &F,
                         const std::vector<MapPoint *> &vpMapPoints,
                         std::vector<MapPoint *> &vpOutMPs,
                         const float th = 3);
  int SearchByProjectionCustom(Frame &F,
                               const std::vector<MapPoint *> &vpMapPoints,
                               std::vector<MapPoint *> &vpOutMPs,
                               const float th = 3);

  // Project MapPoints seen in KeyFrame into the Frame and search matches.
  // Used in relocalisation (Tracking)
  int SearchByProjection(Frame &CurrentFrame, Frame *pKF, const std::set<MapPoint *> &sAlreadyFound,
                         const float th, const float ORBdist);

  virtual int SearchByProjection(Frame &CurrentFrame,
                                 const std::vector<MapPoint *> &vpMapPoints,
                                 const std::set<MapPoint *> &sAlreadyFound,
                                 const float th,
                                 const float ORBdist);

  // // Project MapPoints tracked in last frame into the current frame and search matches.
  // // Used to track from previous frame (Tracking)
  // int SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th, const bool bMono);

  // // Project MapPoints seen in KeyFrame into the Frame and search matches.
  // // Used in relocalisation (Tracking)
  // int SearchByProjection(Frame &CurrentFrame,
  //                        KeyFrame *pKF,
  //                        const std::set<MapPoint *> &sAlreadyFound,
  //                        const float th,
  //                        const int ORBdist);

  // // Project MapPoints using a Similarity Transformation and search matches.
  // // Used in loop detection (Loop Closing)
  // int SearchByProjection(KeyFrame *pKF,
  //                        cv::Mat Scw,
  //                        const std::vector<MapPoint *> &vpPoints,
  //                        std::vector<MapPoint *> &vpMatched,
  //                        int th);

  // // Search matches between MapPoints in a KeyFrame and ORB in a Frame.
  // // Brute force constrained to ORB that belong to the same vocabulary node (at a certain level)
  // // Used in Relocalisation and Loop Detection
  // int SearchByBoW(KeyFrame *pKF, Frame &F, std::vector<MapPoint *> &vpMapPointMatches);
  // int SearchByBoW(KeyFrame *pKF1, KeyFrame *pKF2, std::vector<MapPoint *> &vpMatches12);
  virtual int SearchByBruteForce(const std::vector<MapPoint *> &vpMPs,
                                 Frame &F,
                                 std::vector<MapPoint *> &vpMapPointMatches,
                                 bool useEnhancedMPs = true);
  virtual int SearchByBruteForce(Frame *pKF, Frame &F, std::vector<MapPoint *> &vpMapPointMatches, bool useEnhancedMPs = true);
  int SearchByBruteForce(Frame *pKF, Frame &F, std::vector<std::pair<int, int>> &corres_id, bool useEnhancedMPs);
  int SearchByBoW(Frame *pKF, Frame &F, std::vector<MapPoint *> &vpMapPointMatches, bool useEnhancedMPs = true);
  int SearchByBoW(DescMap *desc_map, Frame &F, std::vector<MapPoint *> &vpMapPointMatches);

  // // Matching for the Map Initialization (only used in the monocular case)
  // int SearchForInitialization(Frame &F1,
  //                             Frame &F2,
  //                             std::vector<cv::Point2f> &vbPrevMatched,
  //                             std::vector<int> &vnMatches12,
  //                             int windowSize = 10);
  //
  // // Matching to triangulate new MapPoints. Check Epipolar Constraint.
  // int SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2, cv::Mat F12,
  //                            std::vector<std::pair<size_t, size_t>> &vMatchedPairs, const bool bOnlyStereo);
  //
  // // Search matches between MapPoints seen in KF1 and KF2 transforming by a Sim3 [s12*R12|t12]
  // // In the stereo and RGB-D case, s12=1
  // int SearchBySim3(KeyFrame *pKF1,
  //                  KeyFrame *pKF2,
  //                  std::vector<MapPoint *> &vpMatches12,
  //                  const float &s12,
  //                  const cv::Mat &R12,
  //                  const cv::Mat &t12,
  //                  const float th);
  //
  // // Project MapPoints into KeyFrame and search for duplicated MapPoints.
  virtual int Fuse(Frame *pF, const std::vector<MapPoint *> &vpMapPoints, const float th = 3.0);
  virtual int FuseNew(Frame *pF, std::vector<MapPoint *> &vpMapPoints, const float th = 3.0);

  // // Project MapPoints into KeyFrame using a given Sim3 and search for duplicated MapPoints.
  // int Fuse(KeyFrame *pKF, cv::Mat Scw, const std::vector<MapPoint *> &vpPoints,
  //          float th, vector<MapPoint *> &vpReplacePoint);

 public:

  static const int TH_LOW;
  static const int TH_HIGH;
  static const int HISTO_LENGTH;

 protected:

  // bool CheckDistEpipolarLine(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &F12, const KeyFrame *pKF);

  float RadiusByViewingCos(const float viewCos);

  void ComputeThreeMaxima(std::vector<int> *histo, const int L, int &ind1, int &ind2, int &ind3);

  float mfNNratio;
  bool mbCheckOrientation;
};

} // namespace relocalization

} // namespace dsl

#endif // DSL_FEATURE_MATCHER_H
