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
// Created by hyye on 8/24/20.
//

#include "util/global_calib.h"
#include "relocalization/desc_map.h"
#include "relocalization/converter.h"
#include "relocalization/feature_matcher.h"
#include "relocalization/sp_matcher.h"
#include "relocalization/reloc_optimization/optimizer.h"
#include "util/timing.h"
#include <opencv2/calib3d.hpp>

// #include <opencv2/flann.hpp>
#include <cmath>
#include <algorithm>

#include <boost/iterator/counting_iterator.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <boost/serialization/unique_ptr.hpp>

#include <queue>

#include "fmt/core.h"
#include "fmt/color.h"

#define USE_FLANN

namespace dsl::relocalization {

typedef nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<float, FLANNPointsWithIndices>, FLANNPointsWithIndices, 2>
    KDTree;

void DescMap::SetFixMap(bool bFixMap) {
  mbFixMap = bFixMap;
}

void DescMap::SetKeyFrameCulling(bool bKFCulling) {
  mbKFCulling = bKFCulling;
}

// Update mvpLocalMapPoints
void DescMap::UpdateLocalMap(const std::vector<PointWithIndex> &points_with_indices) {
  mvpLocalMapPoints.clear();
  std::set<MapPoint *> spMP;

  // for (auto&& pMP : mlpRecentAddedMapPoints) {
  //   spMP.insert(pMP);
  // }

  std::set<Frame *> covisible_frames;

  for (auto &&pwi: points_with_indices) {
    unsigned int global_idx = pwi.index_in_surfel_map;
    if (map_idx_map_points.count(global_idx)) {
      for (auto &&map_point : map_idx_map_points[global_idx]) {
        MapPoint *pMP = map_point;
        pMP->mnTrackReferenceForFrame = mpCurrentFrame->mnId;
        spMP.insert(pMP);

        for (auto &&it: pMP->mObservations) {
          covisible_frames.insert(it.first);
        }
        // mvpLocalMapPoints.push_back(pMP);
      }
    }
  }
  int num_local_bef = spMP.size();
  for (auto &&pCovKF: covisible_frames) {
    for (auto &&pMP: pCovKF->mvpMapPoints) {
      if (pMP && !pMP->isBad()) {
        spMP.insert(pMP);
      }
    }
  }
  int num_local_aft = spMP.size();
  LOG(INFO) << "BEFORE: " << num_local_bef << " AFTER: " << num_local_aft;

  mvpLocalMapPoints.insert(mvpLocalMapPoints.end(), spMP.begin(), spMP.end());
}

// TODO: valid index mask & keypoint + radius point check to replace flann?
/// \brief
/// \param points_with_indices
/// \param frame
/// \param neighbor_indices
void DescMap::SetPointsAndFrame(const std::vector<PointWithIndex> &points_with_indices,
                                Frame *pFrame,
                                std::vector<std::vector<unsigned int>> &neighbor_indices) {
  mpCurrentFrame = pFrame;
  if (!use_superpoint)
    mpCurrentFrame->ComputeBoW();

  pFrame->mvEnhancedMapPointGlobalIds = std::vector<unsigned int>(pFrame->N, 0);

  timing::Timer local_map_timer("map_build.local_map");
  // Update mvpLocalMapPoints by projected indices
  UpdateLocalMap(points_with_indices);
  local_map_timer.Stop();

  // Set pFrame->mvpMapPoints from map projection
  // SearchLocalPoints();
  //
  // for (int f_id = 0; f_id < mpCurrentFrame->mvpMapPoints.size(); ++f_id) {
  //   auto &&pMP = mpCurrentFrame->mvpMapPoints[f_id];
  //   if (pMP) {
  //     pMP->IncreaseFound();
  //     pMP->AddObservation(mpCurrentFrame, f_id);
  //   }
  // }

  const std::vector<cv::KeyPoint> &key_points = pFrame->mvKeysUn;
  const cv::Mat &descs = pFrame->mDescriptors;

  int cnt = 0;
  double sum_min_dist = 0, sum_max_dist = 0;
  int cnt_matches = 0;
  int maxFound = 0;

  // NOTE: find neighbors
  {
    timing::Timer neighbor_timer("map_build.find_neighbor");
    FLANNPointsWithIndices
        pc = FLANNPointsWithIndices(points_with_indices.size(), points_with_indices.data());
    KDTree kdtree = KDTree(2, pc, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    kdtree.buildIndex();

    for (int f_id = 0; f_id < key_points.size(); ++f_id) {
      const cv::KeyPoint &kpt = key_points[f_id];
      const cv::Mat &desc = descs.row(f_id);
      std::vector<unsigned int> neighbor_points_ids;

      float p_query[2] = {kpt.pt.x, kpt.pt.y};
      std::vector<std::pair<size_t, float> > ret_matches;
      const float search_radius = kpt.size / 2 * kpt.size / 2; // since it is L2_Simple
      int num_found = kdtree.radiusSearch(p_query, search_radius, ret_matches, nanoflann::SearchParams());
      if (num_found > maxFound) maxFound = num_found;
      if (num_found > 0) {
        auto comp_matches =
            [](const std::pair<size_t, float> &m1, const std::pair<size_t, float> &m2) {
              return m1.second < m2.second;
            };

        double min_dist = sqrt(std::min_element(ret_matches.begin(), ret_matches.end(), comp_matches)->second);
        double max_dist = sqrt(std::max_element(ret_matches.begin(), ret_matches.end(), comp_matches)->second);

        sum_min_dist += min_dist;
        sum_max_dist += max_dist;
        cnt += 1;
        cnt_matches += num_found;

        for (auto &&ret_match : ret_matches) {
          neighbor_points_ids.push_back(points_with_indices[ret_match.first].index_in_surfel_map);
        }
      }

      neighbor_indices.push_back(neighbor_points_ids);

      if (neighbor_points_ids.size() > 0 && !mpCurrentFrame->mvpMapPoints[f_id] && !mbFixMap) {
        unsigned int closest_global_idx = neighbor_points_ids[0];
        cv::Point3f worldPos = GetGlobalPoint(closest_global_idx);
        if (!map_idx_map_points.count(closest_global_idx)) {
          map_idx_map_points[closest_global_idx] = std::vector<MapPoint *>();
        }
        std::unique_ptr<MapPoint> map_point_ptr = std::make_unique<MapPoint>(cv::Mat(worldPos),
                                                                             this,
                                                                             pFrame,
                                                                             f_id,
                                                                             closest_global_idx);
        ///< call AddMapPoint, since we haven't done it in SearchLocalPoints for these points
        MapPoint *pMP = map_point_ptr.get();
        mlpRecentAddedMapPoints.push_back(pMP);
        pFrame->mvEnhancedMapPointGlobalIds[f_id] = closest_global_idx;
        pFrame->AddMapPoint(pMP, f_id);
        pMP->AddObservation(pFrame, f_id);
        pMP->neighbor_points_global_ids = neighbor_points_ids;
        map_idx_map_points[closest_global_idx].push_back(pMP);
        all_map_points.insert(std::pair<unsigned long, std::unique_ptr<MapPoint>>(pMP->GetId(),
                                                                                  std::move(map_point_ptr)));
      }
    }
  }


  if (!mbFixMap) {
    std::unique_ptr<ORBmatcher> matcher = std::make_unique<ORBmatcher>(0.6, false);
    if (use_superpoint)
      matcher = std::make_unique<SPMatcher>(0.95);
    static int nFusedTotal = 0;

    timing::Timer fuse_timer("map_build.fuse");
    int nFused = matcher->Fuse(mpCurrentFrame, mvpLocalMapPoints);
    fuse_timer.Stop();

    timing::Timer update_timer("map_build.update");
    // Update points
    std::vector<MapPoint *> vpMapPointMatches = mpCurrentFrame->mvpMapPoints;
    for (size_t i = 0, iend = vpMapPointMatches.size(); i < iend; i++) {
      MapPoint *pMP = vpMapPointMatches[i];
      if (pMP) {
        if (!pMP->isBad()) {
          pMP->ComputeDistinctiveDescriptors();
          pMP->UpdateNormalAndDepth();
        }
      }
    }
    mpCurrentFrame->UpdateConnections();
    update_timer.Stop();

    // SearchInNeighbors
    int nn = 20;
    int nFusedKF = 0;
    std::vector<Frame *> vpNeighKFs = mpCurrentFrame->GetBestCovisibilityKeyFrames(nn);
    // WARNING: SKIP SECOND NEIGHBORS

    std::vector<MapPoint *> vpNewMPs;
    // for (size_t i = 0, iend = vpMapPointMatches.size(); i < iend; i++) {
    //   MapPoint *pMP = vpMapPointMatches[i];
    //   if (pMP && pMP->mnFirstKFid == mpCurrentFrame->mnId && pMP->mObservations.size() == 1) {
    //     vpNewMPs.push_back(pMP);
    //   }
    // }
    // for (std::vector<Frame *>::iterator vit = vpNeighKFs.begin(), vend = vpNeighKFs.end(); vit != vend; vit++) {
    //   Frame *pKFi = *vit;
    //   // LOG(INFO) << pKFi->mnId << "???" << vpMapPointMatches.size();
    //   nFusedKF += matcher->FuseNew(pKFi, vpNewMPs);
    // }

    nFusedTotal += nFused + nFusedKF;
    LOG(INFO) << fmt::format(fmt::fg(fmt::color::azure),
                             "nFused: {}, nFusedKF {}, nFusedTotal {}, vpNeighKFs {}, vpNewMPs {}",
                             nFused, nFusedKF, nFusedTotal, vpNeighKFs.size(), vpNewMPs.size());

    timing::Timer culling_timer("map_build.culling");
    if (mbKFCulling) {
      MapPointCulling();
      mpCurrentFrame->UpdateConnections();
      KeyFrameCulling();
    } else {
      MapPointCulling();
    }
  }
  // avoid NULL after Culling
  mvpMapPointMatches = mpCurrentFrame->mvpMapPoints;

  {
    double num_visible = 0, num_found = 0, num_points = 0, num_ratio = 0;
    for (auto &&pMP : mvpMapPointMatches)
      if (pMP) {
        num_visible += pMP->mnVisible;
        num_found += pMP->mnFound;
        num_ratio += pMP->GetFoundRatio();
        num_points += 1;
      }
    LOG(INFO) << fmt::format(fmt::fg(fmt::color::aqua),
                             "avg_nvisible: {:.1f}, avg_nfound: {:.1f}, num_ratio: {:.1f} @ {}",
                             num_visible / num_points, num_found / num_points, num_ratio / num_points,
                             __func__);
  }

  LOG(INFO) << "maxFound: " << maxFound;
  // LOG(INFO) << sum_min_dist / cnt << " - " << sum_max_dist / cnt;
  // LOG(INFO) << double(cnt_matches) / cnt;
  // LOG(INFO) << fmt::format(fmt::emphasis::bold | fg(fmt::color::red), "color!");
}

std::list<MapPoint *>::iterator DescMap::EraseFromRecentMapPoints(MapPoint *pMP) {
  std::list<MapPoint *>::iterator lit = mlpRecentAddedMapPoints.begin();
  while (lit != mlpRecentAddedMapPoints.end()) {
    MapPoint *pRecentMP = *lit;
    if (pRecentMP == pMP) {
      lit = mlpRecentAddedMapPoints.erase(lit);
      break;
    } else
      lit++;
  }
  return lit;
}

void DescMap::EraseMapPoint(MapPoint *pMP) {
  std::vector<MapPoint *> &vMapPoints = map_idx_map_points[pMP->idx_in_surfel_map];
  for (int i = 0; i < vMapPoints.size(); ++i) {
    MapPoint *pMPinMap = vMapPoints[i];
    if (pMPinMap == pMP) {
      std::swap(vMapPoints[i], vMapPoints.back());
      vMapPoints.pop_back();
    }
  }
  if (vMapPoints.empty()) {
    map_idx_map_points.erase(pMP->idx_in_surfel_map);
  }
  all_map_points.erase(pMP->GetId());

  // TODO: This only erase the pointer.
  // Delete the MapPoint
}

void DescMap::EraseKeyFrame(Frame *pF) {
  all_keyframes.erase(pF->mnId);
}

void DescMap::MapPointCulling() {
  int nMPCulled = 0;

  const int cnThObs = 2;
  const unsigned long int nCurrentKFid = mpCurrentFrame->mnId;
  const int cnMaxFidDiff = 2;

  int cntcull = 0;
  // Check Recent Added MapPoints
  LOG(INFO) << "mlpRecentAddedMapPoints: " << mlpRecentAddedMapPoints.size();
  std::list<MapPoint *>::iterator lit = mlpRecentAddedMapPoints.begin();
  while (lit != mlpRecentAddedMapPoints.end()) {
    MapPoint *pMP = *lit;
    if (pMP->isBad()) {
      // FIXME: this should not happen
      lit = mlpRecentAddedMapPoints.erase(lit);
      ++nMPCulled;
    } /*else if (pMP->GetFoundRatio() < 0.25f) {
      pMP->SetBadFlag();
      // lit = mlpRecentAddedMapPoints.erase(lit);  // NOTE: done in SetBadFlag!
      ++nMPCulled;
    }*/ else if (((int) nCurrentKFid - (int) pMP->mnFirstKFid) >= cnMaxFidDiff && pMP->mObservations.size() <= cnThObs) {
      pMP->SetBadFlag();
      lit = mlpRecentAddedMapPoints.erase(lit);
      ++nMPCulled;
    } else if (((int) nCurrentKFid - (int) pMP->mnFirstKFid) >= cnMaxFidDiff + 1)
      lit = mlpRecentAddedMapPoints.erase(lit);
    else
      lit++;
    cntcull++;
  }
  LOG(INFO) << fmt::format("------- nMPCulled {} -------", nMPCulled);
}

void DescMap::KeyFrameCulling() {
  std::vector<Frame *> vpLocalKeyFrames = mpCurrentFrame->mvpOrderedConnectedKeyFrames;

  for (std::vector<Frame *>::iterator vit = vpLocalKeyFrames.begin(), vend = vpLocalKeyFrames.end(); vit != vend;
       vit++) {
    Frame *pKF = *vit;
    if (pKF->mnId == 1)
      continue;
    const std::vector<MapPoint *> vpMapPoints = pKF->mvpMapPoints;

    const int thObs = 3;
    int nRedundantObservations = 0;
    int nMPs = 0;
    for (size_t i = 0, iend = vpMapPoints.size(); i < iend; i++) {
      MapPoint *pMP = vpMapPoints[i];
      if (pMP) {
        if (!pMP->isBad()) {
          nMPs++;
          if (pMP->mObservations.size() > thObs) {
            const int &scaleLevel = pKF->mvKeysUn[i].octave;
            const std::map<Frame *, size_t> observations = pMP->GetObservations();
            int nObs = 0;
            for (std::map<Frame *, size_t>::const_iterator mit = observations.begin(), mend = observations.end();
                 mit != mend; mit++) {
              Frame *pKFi = mit->first;
              if (pKFi == pKF)
                continue;
              const int &scaleLeveli = pKFi->mvKeysUn[mit->second].octave;

              if (scaleLeveli <= scaleLevel + 1) {
                nObs++;
                if (nObs >= thObs)
                  break;
              }
            }
            if (nObs >= thObs) {
              nRedundantObservations++;
            }
          }
        }
      }
    }

    if (nRedundantObservations > 0.9 * nMPs) {
      LOG(INFO) << "------- culling kf ------- " << pKF->mnId;
      pKF->SetBadFlag(this);
    }
  }
}

}