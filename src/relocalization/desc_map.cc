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

#include "util/global_calib.h"
#include "relocalization/desc_map.h"
#include "relocalization/converter.h"
#include "relocalization/feature_matcher.h"
#include "relocalization/sp_matcher.h"
#include "relocalization/pnp_solver.h"
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

namespace dsl {

namespace relocalization {

using namespace DBoW2;

void CvFlannRadiusSearch(std::vector<PointWithIndex> &points_with_indices,
                         std::vector<cv::KeyPoint> &key_points,
                         cv::Mat &descs) {
  cv::Mat_<float> points_db(0, 2);
  for (auto &&p : points_with_indices) {
    cv::Mat row = (cv::Mat_<float>(1, 2) << p.pt.x, p.pt.y);
    points_db.push_back(row);
  }
  cv::flann::Index flann_index(points_db, cv::flann::KDTreeIndexParams(1));

  const unsigned int max_neighbours = 10;

  int cnt = 0;
  int found_cnt = 0;
  double sum = 0;
  for (auto &&kpt : key_points) {
    cv::Mat query = (cv::Mat_<float>(1, 2) << kpt.pt.x, kpt.pt.y);
    std::vector<int> indices;
    std::vector<float> dists;
    float radius = 31.0;
    int num_found = flann_index.radiusSearch(query, indices, dists, radius, max_neighbours,
                                             cv::flann::SearchParams(32));
    if (num_found > 0) {
      found_cnt += num_found;
      double dist = cv::norm(points_with_indices[indices[0]].pt - kpt.pt);
      sum += dist;
      cnt += 1;
      // LOG(INFO) << dist << " " <<  points_with_indices[indices[0]].pt << " " << kpt.pt;
    }
  }
  LOG(INFO) << sum / cnt;
  LOG(INFO) << double(found_cnt) / cnt;
}

void DescMap::SetGlobalVertices(float *vertices, size_t vertex_size, size_t num_vertices) {
  vertices_ = vertices;
  vertex_size_ = vertex_size;
  num_vertices_ = num_vertices;
}

cv::Point3f DescMap::GetGlobalPoint(size_t idx) {
  LOG_ASSERT(idx < num_vertices_);
  LOG_ASSERT(vertex_size_ > 0);

  float x = vertices_[idx * vertex_size_];
  float y = vertices_[idx * vertex_size_ + 1];
  float z = vertices_[idx * vertex_size_ + 2];
  return cv::Point3f(x, y, z);
}

cv::Point3f DescMap::GetGlobalNormal(size_t idx) {
  LOG_ASSERT(idx < num_vertices_) << idx << ": " << num_vertices_;
  LOG_ASSERT(vertex_size_ > 0);

  float nx = vertices_[idx * vertex_size_ + 8];
  float ny = vertices_[idx * vertex_size_ + 9];
  float nz = vertices_[idx * vertex_size_ + 10];
  return cv::Point3f(nx, ny, nz);
}

std::vector<MapPoint *> DescMap::GetMapPoints() {
  return mvpMapPoints;
}

// Calculate mvpMapPointMatches
void DescMap::ComputeLocalBoW(const std::vector<PointWithIndex> &points_with_indices,
                              Frame *pFrame) {
  mpCurrentFrame = pFrame;
  mpCurrentFrame->ComputeBoW();
  UpdateLocalMap(points_with_indices);
  mvpMapPoints = mvpLocalMapPoints;
  mDescriptors = cv::Mat();
  int cnt = 0;
  for (MapPoint *pMP : mvpMapPoints) {
    mDescriptors.push_back(pMP->mDescriptor);
    cnt += 1;
  }
  mBowVec.clear();
  mFeatVec.clear();

  // transform for local points
  std::vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
  mpCurrentFrame->mpORBvocabulary->transform(vCurrentDesc, mBowVec, mFeatVec, 4);

  // LOG(INFO) << "?? " << vCurrentDesc.size() << " " << cnt;

  // LOG(INFO) << mvpMapPoints.size() << " " << mDescriptors.size();

  // LOG(INFO) << mBowVec.size() << " " << mFeatVec.size();
  // for (auto &&nodeIdDescId: mFeatVec) {
  //   std::string out = fmt::format("NodeId: {}, DescId: ", nodeIdDescId.first);
  //   for (auto &&desc_id : nodeIdDescId.second) {
  //     out += fmt::format("{} ", desc_id);
  //   }
  //   LOG(INFO) << out;
  // }

  ORBmatcher matcher(0.75, false);
  std::vector<MapPoint *> vpMapPointMatches;
  int nmatches = matcher.SearchByBoW(this, *mpCurrentFrame, vpMapPointMatches);
  mvpMapPointMatches = vpMapPointMatches;
  LOG(INFO) << "nmatches: " << nmatches;
}

void DescMap::ComputeFrameNodeIds(std::vector<Frame *> &frames, int levelsup) {
  for (Frame *pF : frames) {
    pF->mvFeatNodeId.resize(pF->N);
    for (int idx = 0; idx < pF->N; ++idx) {
      DBoW2::WordId word_id = mpVoc->transform(pF->mDescriptors.row(idx));
      NodeId nid = mpVoc->getParentNode(word_id, levelsup);
      pF->mvFeatNodeId[idx] = nid;
    }
  }
}

void DescMap::SetVLADPath(std::string database_vlad_path, std::string query_vlad_path) {
  std::vector<std::pair<std::string, Frame *>> vPairs;
  for (auto &&item: all_keyframes) {
    vPairs.emplace_back(item.second->mTimeStamp, item.second.get());
  }
  std::sort(vPairs.begin(), vPairs.end());
  std::vector<Frame *> frames;
  for (auto &&pair: vPairs) {
    frames.push_back(pair.second);
  }
  vlad_database_ptr = std::make_unique<VLADDatabase>(database_vlad_path, frames);
  vlad_database_ptr->SetQueryDatabase(query_vlad_path);
}

// TODO: add more contraints to form this graph
void DescMap::CalcConnectedMapPointWeights() {
  for (auto &&item: all_map_points) {
    item.second->mConnectedMapPointWeights.clear();
  }
  for (auto &&id_kf : all_keyframes) {
    auto &&kf = id_kf.second;
    for (auto &&pMP : kf->mvpMapPoints) {
      if (pMP) {
        for (auto &&pMPConnected : kf->mvpMapPoints) {
          if (pMPConnected && pMPConnected != pMP) {
            if (!pMP->mConnectedMapPointWeights.count(pMPConnected->GetId()))
              pMP->mConnectedMapPointWeights[pMPConnected->GetId()] = 1;
            else
              pMP->mConnectedMapPointWeights[pMPConnected->GetId()] += 1;
          }
        }
      }
    }
  }
}

cv::Mat DescMap::CheckFundamentalMat(Frame *pF,
                                     Frame *pKFi,
                                     std::vector<MapPoint *> &vpMapPointMatchesi,
                                     int &nmatches,
                                     int reproj_error) {
  int nmatches_check = 0;
  for (auto &&p: vpMapPointMatchesi) {
    if (p) nmatches_check += 1;
  }
  LOG(INFO) << "nmatches_check bef: " << nmatches_check;

  std::vector<uchar> status_f, status_h, status;
  std::vector<int> vnFFeatId;
  std::vector<cv::Point2f> Fpoints;
  std::vector<cv::Point2f> KFpoints;
  // std::vector<MapPoint *> &vpMapPointMatchesi = vvpMapPointMatches[i];
  for (int f_feat_id = 0; f_feat_id < pF->N; ++f_feat_id) {
    MapPoint *pMP = vpMapPointMatchesi[f_feat_id];
    cv::Point2f ptF = pF->mvKeysUn[f_feat_id].pt;
    cv::Point2f ptKF;
    if (pMP && !pMP->isBad()) {
      if (pMP->mObservations.count(pKFi)) {
        int kf_feat_id = pMP->mObservations[pKFi];
        ptKF = pKFi->mvKeysUn[kf_feat_id].pt;
      } else {
        float u, v;
        if (pKFi->PointToPixel(pMP->GetWorldPos(), u, v)) {
          ptKF = cv::Point2f(u, v);
        } else {
          continue;
        }
      }
      Fpoints.push_back(ptF);
      KFpoints.push_back(ptKF);
      vnFFeatId.push_back(f_feat_id);
    }
  }

  if (Fpoints.size() == 0) {
    LOG(WARNING) << "wrong size = 0";
    return cv::Mat();
  }

  cv::Mat Fmat = cv::findFundamentalMat(Fpoints, KFpoints, cv::FM_RANSAC, reproj_error, 0.99, status_f);
  // cv::Mat Hmat = cv::findHomography(Fpoints, KFpoints, CV_RANSAC, 3, status_h);
  // LOG_ASSERT(status_f.size() == vnFFeatId.size()) << " " << status_f.size() << " " << vnFFeatId.size();

  auto &&ValidStatus = [](const std::vector<uchar> &s) {
    int cnt = 0;
    for (auto &&it: s) {
      if (it > 0) cnt += 1;
    }
    return cnt;
  };
  int num_f = ValidStatus(status_f), num_h = ValidStatus(status_h);
  LOG(INFO) << "num_f: " << num_f << " @@@ num_h: " << num_h;
  if (ValidStatus(status_f) > ValidStatus(status_h)) {
    status = status_f;
  } else {
    status = status_h;
  }

  for (int s_id = 0; s_id < status.size(); ++s_id) {
    if (status[s_id] == 0) {
      int f_feat_id = vnFFeatId[s_id];
      vpMapPointMatchesi[f_feat_id] = static_cast<MapPoint *>(NULL);
      --nmatches;
    }
  }

  nmatches_check = 0;
  for (auto &&p: vpMapPointMatchesi) {
    if (p) nmatches_check += 1;
  }
  LOG(INFO) << "nmatches_check aft: " << nmatches_check;

  return Fmat;
}
bool DescMap::Relocalization(Frame *pFrame) {
  mpCurrentFrame = pFrame;
  if (!use_superpoint)
    mpCurrentFrame->ComputeBoW();

  mpCurrentFrame->mTcw_pnp = cv::Mat();

  // for visualization only
  mvpMapPointMatches.clear();
  mvCandidateKFId.clear();
  mnMatchKeyFrameDBId = 0;

  timing::Timer detect_timer("desc_map.detect");
  std::vector<Frame *> vpCandidateKFs;
  // std::vector<Frame *> vpCandidateKFs = keyframe_db.DetectRelocalizationCandidates(mpCurrentFrame);
  if (config.use_vlad)
    vpCandidateKFs = vlad_database_ptr->DetectRelocalizationCandidates(mpCurrentFrame);
  else
    vpCandidateKFs = keyframe_db.DetectRelocalizationCandidates(mpCurrentFrame);

  // for vis
  mvpDetectedKFs = vpCandidateKFs;

  detect_timer.Stop();
  std::vector<Frame *> vpBestInCandidateKFs;
  std::vector<std::vector<Frame *>> vvpWideCovKFs;
  vpBestInCandidateKFs.resize(vpCandidateKFs.size());
  vvpWideCovKFs.resize(vpCandidateKFs.size());
  LOG(INFO) << "vpCandidateKFs: " << vpCandidateKFs.size() << (config.use_vlad ? " VLAD" : " DBoW");

  if (vpCandidateKFs.empty())
    return false;

  const int nKFs = std::min(int(vpCandidateKFs.size()), config.max_num_kf);

  // We perform first an ORB matching with each candidate
  // If enough matches are found we setup a PnP solver
  std::unique_ptr<ORBmatcher> matcher = std::make_unique<ORBmatcher>(0.75, true);
  if (use_superpoint)
    matcher = std::make_unique<SPMatcher>(0.85);

  std::vector<std::unique_ptr<PnPsolver>> vpPnPsolvers;
  vpPnPsolvers.resize(nKFs);

  std::vector<std::vector<MapPoint *>> vvpMapPointMatches;
  std::vector<std::vector<MapPoint *>> vvpMPRaw;
  std::vector<std::vector<MapPoint *>> vvpMPM3D2D;
  std::vector<std::vector<MapPoint *>> vvpNP;
  vvpMapPointMatches.resize(nKFs);
  vvpMPRaw.resize(nKFs);
  vvpMPM3D2D.resize(nKFs);
  vvpNP.resize(nKFs);

  std::vector<bool> vbDiscarded;
  vbDiscarded.resize(nKFs);

  int nCandidates = 0;
  int bestNMatchesforVis = -1;

  int nNumCovisibles = 5;

  timing::Timer kf_it_timer("desc_map.kf_it");
  for (int i = 0; i < nKFs; ++i) {
    Frame *pKFi = vpCandidateKFs[i];

    std::vector<Frame *> vpCovKFi = pKFi->GetBestCovisibilityKeyFrames(nNumCovisibles);

    // NOTE: no pKFi->isBad();
    mvCandidateKFId.push_back(pKFi->mnId);
    vpCovKFi.push_back(vpCovKFi[0]);
    vpCovKFi[0] = pKFi;

    std::vector<MapPoint *> &vpMapPointMatchesi = vvpMapPointMatches[i];
    std::vector<MapPoint *> &vpMPM3D2Di = vvpMPM3D2D[i];
    std::vector<MapPoint *> &vpMPRawi = vvpMPRaw[i];
    std::vector<MapPoint *> &vpNPi = vvpNP[i];
    std::vector<Frame *> &vpWideCovKFsi = vvpWideCovKFs[i];

    int nmatches = 0;

    // TODO:
    // Add other MapPoints to increase matching?
    // Consider geometric information
    // View point changes?
    // FIXME: matches not enough
    std::vector<std::vector<MapPoint *>> vvpCovKFMatchedMPs;
    std::vector<int> vnCovKFmatches;
    vvpCovKFMatchedMPs.resize(vpCovKFi.size());
    vnCovKFmatches.resize(vpCovKFi.size());
    timing::Timer match_good_timer("desc_map.match_good_points");

    if (!config.use_all_points_in_active_search) {
      if (config.use_covisible_kf_points) {
        int best_nmatches = 0;
        int best_j = 0;

        // std::vector<MapPoint *> vpMPCov;
        // std::set<MapPoint *> spMPCov;
        // for (int j = 0; j < vpCovKFi.size(); ++j) {
        //   if (!vpCovKFi[j] || vpCovKFi[j]->isBad())
        //     continue;
        //   for (auto &&pMP: vpCovKFi[j]->mvpMapPoints) {
        //     if (pMP && !pMP->isBad())
        //       spMPCov.insert(pMP);
        //   }
        // }
        // vpMPCov.insert(vpMPCov.end(), spMPCov.begin(), spMPCov.end());
        //
        // nmatches =
        //     matcher->SearchByBruteForce(vpMPCov, *mpCurrentFrame, vpMapPointMatchesi, false);
        //
        // vpBestInCandidateKFs[i] = vpCandidateKFs[i];

        for (int j = 0; j < vpCovKFi.size(); ++j) {
          if (!vpCovKFi[j] || vpCovKFi[j]->isBad())
            continue;
          int &nmatches_j = vnCovKFmatches[j];
          nmatches_j = matcher->SearchByBruteForce(vpCovKFi[j],
                                                   *mpCurrentFrame,
                                                   vvpCovKFMatchedMPs[j],
                                                   config.use_enhanced_points_in_bf_search);
          if (best_nmatches < nmatches_j) {
            best_nmatches = nmatches_j;
            best_j = j;
          }
        }
        LOG(INFO) << " best_j: " << best_j << " vpCovKFi.size(): " << vpCovKFi.size()
                  << " best_nmatches: " << best_nmatches;
        vpBestInCandidateKFs[i] = vpCovKFi[best_j];
        vpMapPointMatchesi = vvpCovKFMatchedMPs[best_j];

        int cnt_tmp = 0;
        for (auto &&p: vpMapPointMatchesi) {
          if (p) cnt_tmp += 1;
        }
        LOG(INFO) << "cnt_tmp???" << cnt_tmp;

        if (config.use_covisible_kf_points_dbg) {
          for (int j = 0; j < vpCovKFi.size(); ++j) {
            if (!vpCovKFi[j] || vpCovKFi[j]->isBad() || j == best_j) continue;
            for (int fid = 0; fid < vvpCovKFMatchedMPs[j].size(); ++fid) {
              MapPoint *pMP = vpMapPointMatchesi[fid];
              MapPoint *pCovMP = vvpCovKFMatchedMPs[j][fid];
              if ((!pMP || pMP->isBad()) && pCovMP && !pCovMP->isBad()) {
                vpMapPointMatchesi[fid] = pCovMP;
                best_nmatches += 1;
              }
            }
          }
        }
        nmatches = best_nmatches;
        LOG(INFO) << "best: " << nmatches;
      } else {
        vpBestInCandidateKFs[i] = vpCandidateKFs[i];
        // nmatches = matcher.SearchByBoW(pKFi, *mpCurrentFrame, vpMapPointMatchesi, config.use_enhanced_points_in_bf_search);
        nmatches =
            matcher->SearchByBruteForce(pKFi,
                                        *mpCurrentFrame,
                                        vpMapPointMatchesi,
                                        config.use_enhanced_points_in_bf_search);

      }
    } else {
      vpBestInCandidateKFs[i] = vpCandidateKFs[i];
    }

    {
      Frame *pBestKF = vpBestInCandidateKFs[i];
      LOG_ASSERT(pBestKF->mvpOrderedConnectedKeyFrames.size() == pBestKF->mvOrderedWeights.size());
      int num_connected = pBestKF->mvpOrderedConnectedKeyFrames.size();
      double best_weight = pBestKF->mvOrderedWeights[0];
      for (int idx_f = 1; idx_f < num_connected; ++idx_f) {
        if (pBestKF->mvOrderedWeights[idx_f] < 0.95 * best_weight
            && pBestKF->mvOrderedWeights[idx_f] > 0.75 * best_weight) {
          vpWideCovKFsi.push_back(pBestKF->mvpOrderedConnectedKeyFrames[idx_f]);
        }
        if (vpWideCovKFsi.size() >= 5) break;
      }
      LOG(INFO) << "vpWideCovKFsi: " << vpWideCovKFsi.size();
    }

    std::set<MapPoint *> spMPFound;

    for (auto &&pMP: vpMapPointMatchesi) {
      if (pMP && !pMP->isBad()) {
        spMPFound.insert(pMP);
      }
    }

    match_good_timer.Stop();
    vpMPRawi = vpMapPointMatchesi;
    LOG(INFO) << "SearchFromCoVisKFs: " << nmatches;

    // vpMapPointMatchesi = std::vector<MapPoint *>(pFrame->N, static_cast<MapPoint *>(NULL));

    // TODO: from map neighbor points to feature 3D-2D
    if (config.active_search || config.use_all_points_in_active_search) {
      timing::Timer active_search_timer("desc_map.active_search");
      std::set<MapPoint *> neighbor_candidate_points;

      if (config.use_all_points_in_active_search) {
        vpMapPointMatchesi = std::vector<MapPoint *>(pFrame->N, static_cast<MapPoint *>(NULL));
        spMPFound.clear();
      }

      // (S)
      if (config.use_enhanced_points_in_active_search || config.use_all_points_in_active_search) {
        for (MapPoint *pMP: pKFi->mvpMapPoints) {
          if (pMP && !pMP->isBad() && (pMP->mbEnhanced || config.use_all_points_in_active_search)) {
            if (!use_superpoint) {
              if (pMP->node_id == 0) {
                LOG(ERROR) << "pMP->node_id == 0";
                continue;
              }
            }
            if (!spMPFound.count(pMP))
              neighbor_candidate_points.insert(pMP);
          }
        }
      }
      LOG(INFO) << "neighbor_candidate_points w/o neighbor points: " << neighbor_candidate_points.size();

      // (W) wide covisible kfs
      if (config.use_wide_enhanced_points_in_active_search) {
        // vpWideCovKFsi
        for (Frame *pWKF: vpCovKFi) {
          if (pWKF == pKFi) continue;
          for (MapPoint *pMP: pWKF->mvpMapPoints) {
            if (pMP && !pMP->isBad() && !pMP->mbEnhanced)
              if (!spMPFound.count(pMP))
                neighbor_candidate_points.insert(pMP);
          }
        }
      }

      // FIXME: no scores from sp
      // (N)
      if (config.use_good_neighbor_points_in_active_search && !config.use_all_points_in_active_search) {
        // std::vector<MapPoint *> &vMPs = vpMapPointMatchesi; // from valid matches -- less
        std::vector<MapPoint *> &vMPs = pKFi->mvpMapPoints; // from KF's map points -- more
        for (MapPoint *pMP: vMPs) {
          if (pMP && !pMP->isBad() && !pMP->mbEnhanced) {
            std::vector<unsigned int> &neighbor_gids = pMP->neighbor_points_global_ids;
            for (unsigned int gid : neighbor_gids) {
              // map_idx_enhanced_map_points // from points from this frame - NEED TO CHANGE BuildPrioritizedNodes
              // map_idx_map_points // from matched points across frames
              // std::unordered_map<unsigned int, std::vector<MapPoint * >> &globalIdx_vMPs = map_idx_enhanced_map_points;
              std::unordered_map<unsigned int, std::vector<MapPoint * >> &globalIdx_vMPs = map_idx_map_points;
              if (globalIdx_vMPs.count(gid)) {
                if (0 && !use_superpoint) {
                  std::vector<std::pair<int, MapPoint *>> vCostMP;
                  for (MapPoint *pNMP : globalIdx_vMPs[gid]) {
                    LOG_ASSERT(node_cost.count(pNMP->node_id) && pNMP->node_id != 0) << " " << pNMP->node_id;
                    vCostMP.push_back(std::make_pair(node_cost[pNMP->node_id], pNMP));
                  }
                  sort(vCostMP.begin(), vCostMP.end());
                  for (int vid = 0; vid < vCostMP.size(); ++vid) {
                    MapPoint *pNMP = vCostMP[vid].second;
                    neighbor_candidate_points.insert(pNMP);
                  }
                } else {
                  for (MapPoint *pNMP : globalIdx_vMPs[gid]) {
                    neighbor_candidate_points.insert(pNMP);
                  } // for
                } // else
              } // if
            } // for
          } // if
        } // for
      } // if active a

      vpNPi = std::vector<MapPoint *>(neighbor_candidate_points.begin(), neighbor_candidate_points.end());
      LOG(INFO) << "neighbor_candidate_points: " << neighbor_candidate_points.size() << " " << vpNPi.size();

      timing::Timer match_neighbor_timer("desc_map.match_good_neighbor_points");

      {
        int num2D3D = 0;
        std::vector<MapPoint *> vpMP2D3D;
        if (use_superpoint)
          matcher = std::make_unique<SPMatcher>(0.95);
        matcher->SearchByBruteForce(vpNPi,
                                    *mpCurrentFrame,
                                    vpMP2D3D,
                                    config.use_enhanced_points_in_active_search
                                        || config.use_all_points_in_active_search);
        // TODO: rename
        vpMPM3D2Di = vpMP2D3D;
        for (int fid = 0; fid < mpCurrentFrame->N; ++fid) {
          if (!vpMapPointMatchesi[fid] && vpMP2D3D[fid]) {
            vpMapPointMatchesi[fid] = vpMP2D3D[fid];
            ++num2D3D;
          }
        }
        LOG(INFO) << fmt::format(fmt::fg(fmt::color::cyan), "num2D3D: {}", num2D3D);
        nmatches += num2D3D;
      }
      match_neighbor_timer.Stop();
    }

    // Fundamental matrix check
    if (config.fcheck && nmatches >= 8) {
      timing::Timer fcheck_timer("desc_map.fcheck");
      CheckFundamentalMat(pFrame, pKFi, vpMapPointMatchesi, nmatches);

      // if (nmatches >=8)
      //   CheckFundamentalMat(pFrame, pKFi, vpMapPointMatchesi, nmatches, 1);
      LOG(INFO) << "After FCheck: " << nmatches;
    }

    // NOTE: at least show the first one
    if (nmatches > bestNMatchesforVis) {
      bestNMatchesforVis = nmatches;
      mvpCovKFs = vpCovKFi;
      mvpWideCovKFs = vvpWideCovKFs[i]; // vpWideCovKFsi;
      mvpMapPointMatches = vvpMapPointMatches[i];
      mvpMPRaw = vvpMPRaw[i];
      mvpMPM3D2D = vvpMPM3D2D[i];
      mnMatchKeyFrameDBId = vpBestInCandidateKFs[i]->mnId;
    }

    int min_nmatches_to_pnp = 8;
    // if (use_superpoint) min_nmatches_to_pnp = 12;
    if (nmatches < min_nmatches_to_pnp) {
      vbDiscarded[i] = true;
      continue;
    } else {
      std::unique_ptr<PnPsolver> pSolver = std::make_unique<PnPsolver>(*mpCurrentFrame, vvpMapPointMatches[i]);
      pSolver->SetRansacParameters(0.99, 8, 300, 4, 0.4, 5.991 * 3);
      if (use_superpoint) pSolver->SetRansacParameters(0.99, 8, 300, 4, 0.4, 5.991 * 3);
      vpPnPsolvers[i] = std::move(pSolver);
      nCandidates++;
    }
  }
  kf_it_timer.Stop();

  timing::Timer solver_timer("desc_map.solver");
  // Alternatively perform some iterations of P4P RANSAC
  // Until we found a camera pose supported by enough inliers
  bool bMatch = false;

  std::unique_ptr<ORBmatcher> matcher2 = std::make_unique<ORBmatcher>(0.9, true);
  if (use_superpoint)
    matcher2 = std::make_unique<SPMatcher>(1.0);

  int nGood = 0;
  int iKFGood = -1;
  while (nCandidates > 0 && !bMatch) {
    for (int i = 0; i < nKFs; i++) {
      if (vbDiscarded[i])
        continue;

      Frame *pKFi = vpCandidateKFs[i];
      std::vector<Frame *> vpCovKFi = pKFi->GetBestCovisibilityKeyFrames(nNumCovisibles);

      // Perform 5 Ransac Iterations
      std::vector<bool> vbInliers;
      int nInliers;
      bool bNoMore;

      PnPsolver *pSolver = vpPnPsolvers[i].get();
      cv::Mat Tcw = pSolver->iterate(5, bNoMore, vbInliers, nInliers);

      // If Ransac reachs max. iterations discard keyframe
      if (bNoMore) {
        vbDiscarded[i] = true;
        nCandidates--;
      }

      LOG(INFO) << "nKFs: " << i << " mnBestInliers: " << pSolver->mnBestInliers
                << " mnInliersi: " << pSolver->mnInliersi
                << " bNoMore: " << (bNoMore ? "Y" : "N");

      // If a Camera Pose is computed, optimize
      if (!Tcw.empty()) {
        // Tcw.copyTo(mpCurrentFrame->mTcw);
        Tcw.copyTo(mpCurrentFrame->mTcw_pnp);
        Tcw.copyTo(mpCurrentFrame->mTcw_pnp_raw);

        std::set<MapPoint *> sFound;

        std::vector<MapPoint *> vpMPForSearching;
        for (MapPoint *pMP : vpCandidateKFs[i]->mvpMapPoints) {
          if (pMP && !pMP->isBad()) {
            vpMPForSearching.push_back(pMP);
          }
        }
        LOG(INFO) << "before size: " << vpMPForSearching.size();
        for (MapPoint *pMP : vvpMPM3D2D[i]) {  // vvpMPM3D2D[i], vvpNP[i]
          if (pMP && !pMP->isBad()) {
            vpMPForSearching.push_back(pMP);
          }
        }
        LOG(INFO) << "after adding vvpMPM3D2D/2D3D size: " << vpMPForSearching.size();

        const int np = vbInliers.size();

        for (int j = 0; j < np; j++) {
          if (vbInliers[j]) {
            mpCurrentFrame->mvpMapPoints[j] = vvpMapPointMatches[i][j];
            sFound.insert(vvpMapPointMatches[i][j]);
          } else
            mpCurrentFrame->mvpMapPoints[j] = NULL;
        }

        int nGood1 = 0, nGood2 = 0, nGood3 = 0;
        // int nGood = nInliers;
        // WARNING: mpCurrentFrame->mvpMapPoints is used for optimization
        nGood = Optimizer::PoseOptimization(mpCurrentFrame);
        nGood1 = nGood;

        if (nGood < 8)
          continue;

        for (int io = 0; io < mpCurrentFrame->N; io++)
          if (mpCurrentFrame->mvbOutlier[io])
            mpCurrentFrame->mvpMapPoints[io] = static_cast<MapPoint *>(NULL);

        /// If few inliers, search by projection in a coarse window and optimize again
        if (nGood < config.min_nMatched) {
          // TODO: improve search by projection to map points
          // int nadditional = matcher2->SearchByProjection(*mpCurrentFrame, vpCandidateKFs[i], sFound, 10, 100);
          int nadditional;
          if (!use_superpoint)
            nadditional = matcher2->SearchByProjection(*mpCurrentFrame, vpMPForSearching, sFound, 10, 100);
          else
            nadditional =
                matcher2->SearchByProjection(*mpCurrentFrame, vpMPForSearching, sFound, 10, SPMatcher::TH_HIGH * 1.5);

          LOG(INFO) << ">_<? nadditional1 " << nadditional << "? vpMPForSearching: " << vpMPForSearching.size();

          if (nadditional + nGood >= config.min_nGood) {
            // WARNING: NO OPTIMIZATION YET
            // Tcw = pSolver->iterate(5, bNoMore, vbInliers, nInliers);
            // if (!Tcw.empty()) {
            //   Tcw.copyTo(mpCurrentFrame->mTcw_pnp);
            //   nGood = nInliers;
            // }
            nGood = Optimizer::PoseOptimization(mpCurrentFrame);
            nGood2 = nGood;

            /// If many inliers but still not enough, search by projection again in a narrower window
            /// the camera has been already optimized with many points
            if (nGood >= std::min<double>(config.min_nMatched, 0.6 * config.min_nGood) && nGood < config.min_nMatched) {
              sFound.clear();
              for (int ip = 0; ip < mpCurrentFrame->N; ip++)
                if (mpCurrentFrame->mvpMapPoints[ip])
                  sFound.insert(mpCurrentFrame->mvpMapPoints[ip]);
              // nadditional = matcher2->SearchByProjection(*mpCurrentFrame, vpCandidateKFs[i], sFound, 3, 64);
              if (!use_superpoint)
                nadditional = matcher2->SearchByProjection(*mpCurrentFrame, vpMPForSearching, sFound, 3, 64);
              else
                nadditional =
                    matcher2->SearchByProjection(*mpCurrentFrame, vpMPForSearching, sFound, 3, SPMatcher::TH_HIGH);
              LOG(INFO) << ">_<? nadditional2 " << nadditional;

              // Final optimization
              if (nGood + nadditional >= config.min_nMatched) {
                // WARNING: NO OPTIMIZATION YET
                // Tcw = pSolver->iterate(5, bNoMore, vbInliers, nInliers);
                // if (!Tcw.empty()) {
                //   Tcw.copyTo(mpCurrentFrame->mTcw_pnp);
                //   nGood = nInliers;
                // }
                nGood = Optimizer::PoseOptimization(mpCurrentFrame);

                nGood3 = nGood;

                for (int io = 0; io < mpCurrentFrame->N; io++)
                  if (mpCurrentFrame->mvbOutlier[io])
                    mpCurrentFrame->mvpMapPoints[io] = NULL;
              }
            }
          }
        }
        LOG(INFO) << fmt::format("nGood: {}, nGood1: {}, nGood2: {}, nGood3: {}", nGood, nGood1, nGood2, nGood3);
        // /// If the pose is supported by enough inliers stop ransacs and continue
        // if (nGood >= 50) {
        //   bMatch = true;
        //   break;
        // }
        static int sum_nGood = 0;
        static int sum_nGood_cnt = 0;
        if (nGood >= config.min_nGood) {
          sum_nGood += nGood;
          sum_nGood_cnt += 1;
          bMatch = true;
          /// just for visualization
          mvpMapPointMatches = mpCurrentFrame->mvpMapPoints;
          mvpCovKFs = vpCovKFi;
          mvpWideCovKFs = vvpWideCovKFs[i];
          mvpMPRaw = vvpMPRaw[i];
          mvpMPM3D2D = vvpMPM3D2D[i];
          mnMatchKeyFrameDBId = vpBestInCandidateKFs[i]->mnId;
          iKFGood = i;
          break;
        }
        LOG(INFO) << "mean nGood true: " << double(sum_nGood) / sum_nGood_cnt;
      }
    }
  }

  solver_timer.Stop();

  LOG(INFO) << "nGood: " << nGood << " iKFGood: " << iKFGood;

  if (!bMatch) {
    return false;
  } else {
    // mnLastRelocFrameId = mCurrentFrame.mnId;
    return true;
  }

}
// TODO: directly found from map points!!!
void DescMap::BuildPrioritizedNodes(int levelsup) {
  std::vector<MapPoint *> vpMPs;
  // enhanced_map_points
  // all_map_points
  for (auto &&it: all_map_points) {
    vpMPs.push_back(it.second.get());
  }

  if (config.use_enhanced_points_in_active_search || config.use_all_points_in_active_search) {
    for (auto &&it: enhanced_map_points) {
      vpMPs.push_back(it.second.get());
    }
  }

  // n_D(w)
  node_cost.clear();
  std::map<WordId, std::vector<MapPoint *>> nid_vMPs;
  for (MapPoint *pMP: vpMPs) {
    if (!pMP || pMP->isBad()) continue;
    // TODO: to higher level?
    NodeId nid;
    mpVoc->transformToNodeId(pMP->mDescriptor, nid, levelsup);
    pMP->node_id = nid;
    if (!nid_vMPs.count(nid)) {
      nid_vMPs[nid] = std::vector<MapPoint *>();
    }
    nid_vMPs[nid].push_back(pMP);
  }
  std::vector<std::pair<int, WordId> > vPairs;
  for (auto &&it: nid_vMPs) {
    int nid = it.first;
    int cost = it.second.size();
    vPairs.push_back(std::make_pair(cost, nid));
  }
  std::sort(vPairs.begin(), vPairs.end());
  std::list<NodeId> lNids;
  std::list<int> lCs;
  for (size_t i = 0, iend = vPairs.size(); i < iend; i++) {
    node_cost.insert(std::make_pair(vPairs[i].second, vPairs[i].first));
    lCs.push_back(vPairs[i].first);
    lNids.push_back(vPairs[i].second);
  }
  std::list<int>::iterator cost = lCs.begin();
  std::list<NodeId>::iterator nid = lNids.begin();
  // for (; cost != lCs.end(); ++cost, ++nid) {
  //   LOG(INFO) << "cost: " << *cost;
  //   LOG(INFO) << "nid: " << *nid << (mpVoc->GetNodes()[*nid].isLeaf() ? " leaf" : " not leaf");
  // }
  LOG(INFO) << "group size: " << lCs.size();
}

void DescMap::SaveMap(const std::string filename) {
  // NOTE: We will use the covisibilty graph now
  // CalcConnectedMapPointWeights();

  std::ofstream out(filename, std::ios_base::binary);
  if (!out) {
    LOG(ERROR) << "Cannot Write to Mapfile: " << filename;
  }
  LOG(INFO) << "Saving Mapfile: " << filename << std::flush;
  boost::archive::binary_oarchive oa(out, boost::archive::no_header);

  for (auto &&id_kf: all_keyframes) id_kf.second->UpdateConnections();

  std::vector<Frame> keyframes;
  for (auto &&id_kf: all_keyframes) {
    // map points to map of <feat_id, map_point_id>
    id_kf.second->PrepareForSaving();
    keyframes.push_back(*id_kf.second);
  }

  std::vector<MapPoint> map_points;
  for (auto &&idx_map_points: map_idx_map_points) {
    for (auto &&map_point : idx_map_points.second) {
      // observations to map of <frame_id, feat_id>
      map_point->PrepareForSaving();
      map_points.push_back(*map_point);
    }
  }
  oa << map_points;
  oa << keyframes;
  LOG(INFO) << "map_points: " << map_points.size() << " keyframes: " << keyframes.size();
  LOG(INFO) << " ...done" << std::endl;
  out.close();
}

bool DescMap::LoadMap(const std::string filename) {
  std::ifstream in(filename, std::ios_base::binary);
  if (!in) {
    LOG(ERROR) << "Cannot Open Mapfile: " << filename << " , Create a new one";
    return false;
  }
  LOG(INFO) << "Loading Mapfile: " << filename;
  boost::archive::binary_iarchive ia(in, boost::archive::no_header);
  map_idx_map_points.clear();

  std::vector<MapPoint> map_points;
  std::vector<Frame> keyframes;
  ia >> map_points;
  ia >> keyframes;

  for (auto &&map_point:map_points) {
    if (!map_idx_map_points.count(map_point.idx_in_surfel_map)) {
      map_idx_map_points[map_point.idx_in_surfel_map] = std::vector<MapPoint *>();
    }
    std::unique_ptr<MapPoint> map_point_ptr = std::make_unique<MapPoint>(map_point);
    MapPoint *pMP = map_point_ptr.get();
    pMP->mpMap = this;
    all_map_points.insert(std::pair<unsigned long, std::unique_ptr<MapPoint>>(pMP->GetId(), std::move(map_point_ptr)));
    map_idx_map_points[map_point.idx_in_surfel_map].push_back(pMP);
  }

  for (auto &&kf : keyframes) {
    std::unique_ptr<Frame> kf_ptr = std::make_unique<Frame>(kf);
    all_keyframes.insert(std::pair<unsigned long, std::unique_ptr<Frame>>(kf.mnId, std::move(kf_ptr)));
  }

  for (auto &&it: all_map_points) it.second->Restore(all_keyframes);

  for (auto &&it: all_keyframes) it.second->Restore(all_map_points);

  for (auto &&id_kf: all_keyframes) id_kf.second->UpdateConnections();

  // int max_connected = -1;
  // for (auto &&it : map_idx_map_points) {
  //   for (auto &&mp: it.second) {
  //     for (auto &&w : mp->mConnectedMapPointWeights) {
  //       if (w.second > max_connected) max_connected = w.second;
  //     }
  //   }
  // }
  // LOG(INFO) << "max_connected: " << max_connected;

  LOG(INFO) << "map_points: " << map_points.size() << " keyframes: " << keyframes.size();

  LOG(INFO) << "Map Reconstructing not implemented" << std::flush;
  in.close();
  return true;
}

void DescMap::EnhanceKeyframePoints() {
  for (auto &&it : all_keyframes) {
    Frame *pF = it.second.get();
    for (int f_id = 0; f_id < pF->N; ++f_id) {
      int closest_global_idx = pF->mvEnhancedMapPointGlobalIds[f_id];
      if ((pF->mvpMapPoints[f_id] == NULL || config.use_all_points_in_active_search) && closest_global_idx != 0) {
        cv::Point3f worldPos = GetGlobalPoint(closest_global_idx);
        std::unique_ptr<MapPoint> map_point_ptr = std::make_unique<MapPoint>(cv::Mat(worldPos),
                                                                             this,
                                                                             pF,
                                                                             f_id,
                                                                             closest_global_idx);

        MapPoint *pMP = map_point_ptr.get();
        pMP->mbEnhanced = true;
        pF->AddMapPoint(pMP, f_id);
        pMP->AddObservation(pF, f_id);
        enhanced_map_points.insert(std::pair<unsigned long, std::unique_ptr<MapPoint>>(pMP->GetId(),
                                                                                       std::move(map_point_ptr)));
        if (!map_idx_enhanced_map_points.count(closest_global_idx)) {
          map_idx_enhanced_map_points[closest_global_idx] = std::vector<MapPoint *>();
        }
        map_idx_enhanced_map_points[closest_global_idx].push_back(pMP);
      }
    }
  }
}

bool VerifyDescMap(DescMap &desc_map1, DescMap &desc_map2) {

  for (auto &&it: desc_map1.all_keyframes) {
    unsigned long frame_id1 = it.first;
    for (int idx1 = 0; idx1 < desc_map1.all_keyframes[frame_id1]->N; ++idx1) {
      MapPoint *pMP1 = desc_map1.all_keyframes[frame_id1]->GetMapPoint(idx1);
      if (pMP1) {
        if (!pMP1->isBad()) {
          int connected_kf_diff = desc_map1.all_keyframes[frame_id1]->GetConnectedKeyFrames().size()
              - desc_map2.all_keyframes[frame_id1]->GetConnectedKeyFrames().size();
          int desc_diff = cv::norm(desc_map1.all_keyframes[frame_id1]->GetMapPoint(idx1)->GetDescriptor()
                                       - desc_map2.all_keyframes[frame_id1]->GetMapPoint(idx1)->GetDescriptor());
          if (desc_diff != 0 || connected_kf_diff != 0) {
            LOG(ERROR) << "connected_kf_diff: " << connected_kf_diff
                       << std::endl << desc_map1.all_keyframes[frame_id1]->GetConnectedKeyFrames().size()
                       << std::endl << desc_map2.all_keyframes[frame_id1]->GetConnectedKeyFrames().size();
            LOG(ERROR) << "desc_diff: " << desc_diff
                       << std::endl << desc_map1.all_keyframes[frame_id1]->GetMapPoint(idx1)->GetDescriptor()
                       << std::endl << desc_map2.all_keyframes[frame_id1]->GetMapPoint(idx1)->GetDescriptor();
            return false;
          }
        } else {
          LOG(ERROR) << "THIS SHOULD NEVER HAPPEN!";
          return false;
        }
      }
    }
  }

  for (auto &&it: desc_map1.all_map_points) {
    unsigned long mpid1 = it.first;
    int obs_diff = desc_map1.all_map_points[mpid1]->GetObservations().size()
        - desc_map2.all_map_points[mpid1]->GetObservations().size();
    if (obs_diff != 0) {
      LOG(ERROR) << "obs_diff: " << obs_diff;
      return false;
    }

    for (auto &&obs1: desc_map1.all_map_points[mpid1]->GetObservations()) {
      unsigned long frame_id1 = obs1.first->mnId;
      int feat_id_diff =
          desc_map1.all_map_points[mpid1]->GetObservations().at(desc_map1.all_keyframes[frame_id1].get())
              - desc_map2.all_map_points[mpid1]->GetObservations().at(desc_map2.all_keyframes[frame_id1].get());
      if (feat_id_diff != 0) {
        LOG(ERROR) << "feat_id_diff: " << feat_id_diff;
        return false;
      }
    }
  }

  LOG(INFO) << "desc_map1: map_points " << desc_map1.all_map_points.size() << ", keyframes "
            << desc_map1.all_keyframes.size();
  LOG(INFO) << "desc_map2: map_points " << desc_map2.all_map_points.size() << ", keyframes "
            << desc_map2.all_keyframes.size();

  return true;
}

} // namespace relocalization

}  // namespace dsl
