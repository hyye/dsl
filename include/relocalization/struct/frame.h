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

#ifndef DSL_FRAME_H
#define DSL_FRAME_H

#include "util/util_common.h"
#include "relocalization/feature_extractor.h"
#include "relocalization/vocabulary_binary.h"

namespace dsl {

namespace relocalization {

#define FRAME_GRID_ROWS 48
#define FRAME_GRID_COLS 64

class Frame;
class MapPoint;
class DescMap;

class Frame {
 public:
  Frame() {}
  Frame(const cv::Mat &imageIn, const std::string timeStamp, ORBextractor *extractor, OrbVocabularyBinary *voc);

  void ExtractORB(int flag, const cv::Mat &im);
  void ComputeBoW();
  void AddMapPoint(MapPoint *pMP, const size_t idx);  // this is only needed for KeyFrame
  void ReplaceMapPointMatch(const size_t idx, MapPoint *pMP);
  void EraseMapPointMatch(const size_t idx);  // not truly erased

  bool IsInFrustum(const cv::Mat &P, float &u, float &v);
  bool PointToPixel(const cv::Mat &P, float &u, float &v);
  // Check if a MapPoint is in the frustum of the camera
  // and fill variables of the MapPoint to be used by the tracking
  bool IsInFrustum(MapPoint *pMP, float viewingCosLimit);
  bool IsInFrustum(MapPoint *pMP, cv::Mat &Tcw, float viewingCosLimit);

  bool IsInImage(const float x, const float y) const;

  MapPoint *GetMapPoint(const size_t idx);

  void SetPose(cv::Mat Tcw);
  cv::Mat GetPose() const;
  cv::Mat GetPoseInverse() const;
  cv::Mat GetRotation() const;
  cv::Mat GetTranslation() const;
  void UpdatePoseMatrices();

  // Relocalization
  // Frame is abused, replaced with KF?
  std::set<Frame *> GetConnectedKeyFrames();
  std::vector<Frame *> GetBestCovisibilityKeyFrames(const int nKFs);
  // TODO: to update connected keyframeWeights, orderedConnectedKeyframes and orderedWeights
  void UpdateConnections();
  void EraseConnection(Frame *pF);
  void AddConnection(Frame *pF, const int weight);
  void UpdateBestCovisibles();

  void SetBadFlag(DescMap *pMap);

  // Returns the camera center.
  inline cv::Mat GetCameraCenter() {
    return mOw.clone();
  }

  std::vector<size_t> GetFeaturesInArea(const float &x,
                                        const float &y,
                                        const float &r,
                                        const int minLevel = -1,
                                        const int maxLevel = -1) const;

  bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);
  void AssignFeaturesToGrid();

  bool isBad() { return mbBad; }

  bool mbBad = false;

  static long unsigned int nNextId;
  long unsigned int mnId;

  // Frame timestamp
  std::string mTimeStamp;

  // Number of KeyPoints.
  int N = 0;

  // Vector of keypoints (undistorted), we use the undistorted images
  std::vector<cv::KeyPoint> mvKeysUn;

  // Bag of Words Vector structures.
  DBoW2::BowVector mBowVec;
  DBoW2::FeatureVector mFeatVec;

  std::vector<DBoW2::NodeId> mvFeatNodeId;

  // ORB descriptor, each row associated to a keypoint.
  cv::Mat mDescriptors;

  int mnRelocWords = 0;
  int mnRelocQuery = -1;
  float mRelocScore = 0;

  // From N = mvKeysUn.size()
  std::vector<MapPoint *> mvpMapPoints;
  std::vector<unsigned int> mvEnhancedMapPointGlobalIds;

  // Flag to identify outlier associations (used by optimization)
  std::vector<bool> mvbOutlier;

  // Scale pyramid info -- from feature extractor
  int mnScaleLevels;
  float mfScaleFactor;
  float mfLogScaleFactor;
  std::vector<float> mvScaleFactors;
  std::vector<float> mvInvScaleFactors;
  std::vector<float> mvLevelSigma2;
  std::vector<float> mvInvLevelSigma2;

  // Feature extractor
  ORBextractor *mpORBextractor;
  // Vocabulary used for relocalization
  OrbVocabularyBinary *mpORBvocabulary;

  cv::Mat mTcw_pnp;
  cv::Mat mTcw_pnp_raw;

  // Camera pose.
  cv::Mat mTcw;
  cv::Mat mTwc;

  // Rotation, translation and camera center
  cv::Mat mRcw;
  cv::Mat mtcw;
  cv::Mat mRwc;
  cv::Mat mOw; //==mtwc

  // Keypoints are assigned to cells in a grid to reduce matching complexity when projecting MapPoints.
  static float mfGridElementWidthInv;
  static float mfGridElementHeightInv;
  std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];

  std::map<Frame *, int> mConnectedKeyFrameWeights;
  std::vector<Frame *> mvpOrderedConnectedKeyFrames;
  std::vector<int> mvOrderedWeights;

  // Calibration matrix and OpenCV distortion parameters.
  static float fx;
  static float fy;
  static float cx;
  static float cy;
  static float invfx;
  static float invfy;
  // Undistorted Image Bounds (computed once).
  static float mnMinX;
  static float mnMaxX;
  static float mnMinY;
  static float mnMaxY;

  static bool mbInitialComputations;

 private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive &ar, const unsigned int version);

 public:
  // <feat_id, map point id>
  std::map<int, long unsigned int> mFeatIdValidMapPointId;

  void PrepareForSaving();
  void Restore(const std::unordered_map<unsigned long, std::unique_ptr<MapPoint>> &all_map_points);
};

}  // namespace relocalization

}  // namespace dsl

#endif // DSL_FRAME_H
