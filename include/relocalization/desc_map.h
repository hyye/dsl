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

#ifndef DSL_DESC_MAP_H
#define DSL_DESC_MAP_H

#include <opencv2/features2d.hpp>
#include <unordered_map>
#include <Eigen/Core>
#include "util/util_common.h"
#include "relocalization/relocalization_struct.h"
#include "relocalization/struct/key_frame_database.h"
#include "relocalization/struct/vlad_database.h"
#include "relocalization/struct/motion_estimator.h"
#include "relocalization/relocalization_config.h"
#include "nanoflann.hpp"

namespace dsl {

namespace relocalization {

struct PointWithIndex {
  PointWithIndex() {}
  PointWithIndex(const PointWithIndex &p) : pt(p.pt), index_in_surfel_map(p.index_in_surfel_map), dist(p.dist) {}
  PointWithIndex(float x, float y, unsigned int id, float _dist = 0) : pt(x, y), index_in_surfel_map(id), dist(_dist) {}
  cv::Point2f pt;
  unsigned int index_in_surfel_map;
  float dist; /// from distance map
};

struct KeyPointWithDescriptor {
  cv::KeyPoint kpt;
  cv::Mat descriptor;
};

struct FLANNPointsWithIndices {
  inline FLANNPointsWithIndices() {
    num = 0;
    points = 0;
  }
  inline FLANNPointsWithIndices(int n, const PointWithIndex *p) : num(n), points(p) {}
  int num;
  const PointWithIndex *points;
  inline size_t kdtree_get_point_count() const { return num; }

  inline float kdtree_get_pt(const size_t idx, int dim) const {
    if (dim == 0)
      return points[idx].pt.x;
    else
      return points[idx].pt.y;
  }
  template<class BBOX>
  bool kdtree_get_bbox(BBOX & /* bb */) const {
    return false;
  }
};

struct FLANNCvPoint2f {
  inline FLANNCvPoint2f() {
    num = 0;
    points = 0;
  }
  inline FLANNCvPoint2f(int n, const cv::Point2f *p) : num(n), points(p) {}
  int num;
  const cv::Point2f *points;
  inline size_t kdtree_get_point_count() const { return num; }

  inline float kdtree_get_pt(const size_t idx, int dim) const {
    if (dim == 0)
      return points[idx].x;
    else
      return points[idx].y;
  }
  template<class BBOX>
  bool kdtree_get_bbox(BBOX & /* bb */) const {
    return false;
  }
};

class DescMap {
 public:
  void SetGlobalVertices(float *vertices, size_t vertex_size, size_t num_vertices);
  void SetPointsAndFrame(const std::vector<PointWithIndex> &points_with_indices,
                         Frame *pFrame,
                         std::vector<std::vector<unsigned int>> &neighbor_indices);
  void ComputeLocalBoW(const std::vector<PointWithIndex> &points_with_indices, Frame *pFrame);
  void ComputeFrameNodeIds(std::vector<Frame *> &frames, int levelsup);
  std::vector<MapPoint *> GetMapPoints();
  cv::Point3f GetGlobalPoint(size_t idx);
  cv::Point3f GetGlobalNormal(size_t idx);

  void UpdateLocalMap(const std::vector<PointWithIndex> &points_with_indices);
  void SearchLocalPoints();
  void SearchPnPLocalPoints(Frame *pFrame, const std::vector<PointWithIndex> &points_with_indices);
  bool SearchByObservations(Frame *pFrame,
                            Frame *pKF,
                            std::vector<std::pair<cv::KeyPoint, cv::KeyPoint>> &kpt_pairs,
                            Eigen::Matrix3d &R,
                            Eigen::Vector3d &t);

  void SaveMap(const std::string filename);
  bool LoadMap(const std::string filename);

  void EraseMapPoint(MapPoint *pMP);
  std::list<MapPoint *>::iterator EraseFromRecentMapPoints(MapPoint *pMP);

  void EraseKeyFrame(Frame *pF);

  void MapPointCulling();
  void KeyFrameCulling();

  void SetFixMap(bool bFixMap);
  void SetKeyFrameCulling(bool bKFCulling);

  void EnhanceKeyframePoints();

  void BuildPrioritizedNodes(int levelsup = 4);

  void CalcConnectedMapPointWeights();

  bool Relocalization(Frame *pFrame);

  void SetVocabularyBinary(OrbVocabularyBinary *pVoc) { mpVoc = pVoc; }

  void DoOptimization(bool run_opt = true);

  bool ValidateByRenderedDistMap(cv::Mat &dist_img_f);

  KeyFrameDatabase keyframe_db;
  std::unique_ptr<VLADDatabase> vlad_database_ptr;
  void SetVLADPath(std::string vlad_path, std::string query_vlad_path);

  cv::Mat CheckFundamentalMat(Frame *pF,
                              Frame *pKFi,
                              std::vector<MapPoint *> &vpMapPointMatchesi,
                              int &nmatches,
                              int reproj_error = 3);

  bool ValidatePnPByEssentialMat(Frame *pF,
                                 Frame *pKFi,
                                 std::vector<std::pair<cv::KeyPoint, cv::KeyPoint>> &kpt_pairs);

  OrbVocabularyBinary *mpVoc;

  float *vertices_ = 0;
  size_t vertex_size_ = 0;
  size_t num_vertices_ = 0;

  std::map<unsigned long, std::unique_ptr<Frame>> all_keyframes;

  // global idx -> vector of pMP
  std::unordered_map<unsigned int, std::vector<MapPoint * >> map_idx_map_points;
  // mnId -> unique_ptr of MapPoint
  std::unordered_map<unsigned long, std::unique_ptr<MapPoint>> all_map_points;

  std::unordered_map<unsigned long, std::unique_ptr<MapPoint>> enhanced_map_points;
  std::unordered_map<unsigned int, std::vector<MapPoint * >> map_idx_enhanced_map_points;

  std::map<DBoW2::NodeId, int> node_cost;

  // matches for the current frame
  std::vector<Frame *> mvpCovKFs;
  std::vector<Frame *> mvpWideCovKFs;
  std::vector<Frame *> mvpDetectedKFs;
  std::vector<MapPoint *> mvpMapPointMatches;
  std::vector<MapPoint *> mvpMPM3D2D;
  std::vector<MapPoint *> mvpMPRaw;
  unsigned long mnMatchKeyFrameDBId = 0;
  std::vector<unsigned long> mvCandidateKFId;

  std::list<MapPoint *> mlpRecentAddedMapPoints;

  // Current Frame
  Frame *mpCurrentFrame;

  // local map matches for the current frame
  std::vector<MapPoint *> mvpLocalMapPoints;
  std::vector<MapPoint *> mvLocalAvailablePoints;

  // relocalisation info
  unsigned int mnLastRelocFrameId;

  DescMapConfig config;

  bool mbFixMap = false;
  bool mbKFCulling = true;

  // WARNING: it is not reasonable to use Bow for a map
  std::vector<MapPoint *> mvpMapPoints;
  cv::Mat mDescriptors;
  // Bag of Words Vector structures.
  DBoW2::BowVector mBowVec;
  DBoW2::FeatureVector mFeatVec;
};

bool VerifyDescMap(DescMap &desc_map1, DescMap &desc_map2);

} // namespace relocalization

} // namespace dsl

#endif // DSL_DESC_MAP_H
