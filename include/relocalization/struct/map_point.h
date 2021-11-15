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
// Created by hyye on 7/6/20.
//

#ifndef DSL_MAP_POINT_H
#define DSL_MAP_POINT_H

#include <opencv2/core.hpp>
#include "util/util_common.h"
#include "relocalization/boost_archiver.h"

namespace dsl {

namespace relocalization {

class Frame;
class MapPoint;
class DescMap;

class MapPoint {
 public:
  MapPoint();
  MapPoint(const cv::Mat &Pos, DescMap *pMap, Frame *pFrame, const int idxF, const int _index_global);
  virtual ~MapPoint() {}

  bool isBad();
  void SetBadFlag();

  cv::Mat GetWorldPos();
  cv::Mat GetDescriptor() { return mDescriptor; }
  float GetMinDistanceInvariance();
  float GetMaxDistanceInvariance();
  cv::Mat GetNormal();

  int PredictScale(const float currentDist, Frame *pF);

  unsigned int idx_in_surfel_map = 0;

  long int mnFirstKFid = -1;

  // Variables used by the tracking
  float mTrackProjX;
  float mTrackProjY;
  bool mbTrackInView = false;
  int mnTrackScaleLevel;
  float mTrackViewCos;
  long unsigned int mnTrackReferenceForFrame;
  long unsigned int mnLastFrameSeen;

  void IncreaseVisible(int n = 1);
  void IncreaseFound(int n = 1);

  float GetFoundRatio();

  void AddObservation(Frame *pF, size_t idx);
  void EraseObservation(Frame *pF);
  std::map<Frame *, size_t> &GetObservations();

  bool IsInKeyFrame(Frame *pF);

  void Replace(MapPoint *pMP);

  void UpdateNormalAndDepth();
  void ComputeDistinctiveDescriptors();
  long unsigned int GetId();

  // TODO: map point attribute
  bool mbOnPlane = false;

  bool mbEnhanced = false;
  DBoW2::NodeId node_id = 0;

  // Tracking counters
  int mnVisible;
  int mnFound;

  std::vector<unsigned int> neighbor_points_global_ids;

  std::map<unsigned long, int> mConnectedMapPointWeights;

 protected:
  long unsigned int mnId;
  static long unsigned int nNextId;
  // Bad flag (we do not currently erase MapPoint from memory)
  bool mbBad = false;
  MapPoint *mpReplaced;

  // frames observing the point and associated index in keyframe
  // we do not save it in serialization
  std::map<Frame *, size_t> mObservations;

  // TODO: pnp - Position in absolute coordinates
  cv::Mat mWorldPos;
  cv::Mat mDescriptor;

  // Mean viewing direction
  cv::Mat mNormalVector;

  // Scale invariance distances;
  float mfMinDistance; ///< can be observed in  pyramid nLevels -1 (top)
  float mfMaxDistance; ///< can be observed in pyramid 0 (original size)

  DescMap *mpMap;
 private:
  friend class DescMap;
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive &ar, const unsigned int version);

 public:
  std::map<long unsigned int, size_t> mFrameIdFeatId;

  void PrepareForSaving();
  void Restore(const std::map<unsigned long, std::unique_ptr<Frame>> &all_keyframes);
};

}  // namespace relocalization

}  // namespace dsl

#endif // DSL_MAP_POINT_H
