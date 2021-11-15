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

namespace dsl::relocalization {

typedef nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<float, FLANNPointsWithIndices>, FLANNPointsWithIndices, 2>
    KDTree;

void NanoflannKnn(std::vector<PointWithIndex> &points_with_indices,
                  std::vector<cv::KeyPoint> &key_points,
                  cv::Mat &descs) {
  // TODO: nanoflann to asscoiate points
  FLANNPointsWithIndices pc = FLANNPointsWithIndices(points_with_indices.size(), points_with_indices.data());
  KDTree kdtree = KDTree(2, pc, nanoflann::KDTreeSingleIndexAdaptorParams(5));
  kdtree.buildIndex();
  nanoflann::KNNResultSet<float, int, int> result_set(1);

  int ret_index[1];
  float ret_dist[1];

  int cnt = 0;
  double sum = 0;
  for (auto &&kpt : key_points) {
    result_set.init(ret_index, ret_dist);
    Eigen::Vector2f p_query(kpt.pt.x, kpt.pt.y);
    kdtree.findNeighbors(result_set, (float *) &p_query, nanoflann::SearchParams());
    double dist = cv::norm(points_with_indices[ret_index[0]].pt - kpt.pt);
    sum += dist;
    cnt += 1;
    // LOG(INFO) << dist << " " <<  points_with_indices[ret_index[0]].pt << " " << kpt.pt;
  }
  LOG(INFO) << sum / cnt;
}

bool DescMap::ValidatePnPByEssentialMat(Frame *pF, Frame *pKFi,
                                        std::vector<std::pair<cv::KeyPoint, cv::KeyPoint>> &kpt_pairs) {
  ORBmatcher matcher(0.75, true);

  std::vector<std::pair<int, int>> corres_id;

  int nmatches = matcher.SearchByBruteForce(pKFi, *pF, corres_id, config.use_enhanced_points_in_bf_search);
  std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> corres;

  kpt_pairs.clear();

  for (auto &&pair:corres_id) {
    int fid_f = pair.first;
    int fid_kf = pair.second;
    cv::Point2f ptF = pF->mvKeysUn[fid_f].pt;
    cv::Point2f ptKF = pKFi->mvKeysUn[fid_kf].pt;
    float xF = (ptF.x - pF->cx) / pF->fx;
    float yF = (ptF.y - pF->cy) / pF->fy;
    float xKF = (ptKF.x - pKFi->cx) / pKFi->fx;
    float yKF = (ptKF.y - pKFi->cy) / pKFi->fy;
    corres.push_back(std::make_pair(Eigen::Vector3d(xF, yF, 1), Eigen::Vector3d(xKF, yKF, 1)));
    kpt_pairs.push_back(std::make_pair(pF->mvKeysUn[fid_f], pKFi->mvKeysUn[fid_kf]));
  }
  Eigen::Matrix3d R;
  Eigen::Vector3d t;
  // from pKFi to pF
  LOG(INFO) << "corres: " << corres.size();
  cv::Mat mask;
  bool rel_flag = MotionEstimator::solveRelativeRT(corres, R, t, mask);

  std::vector<std::pair<cv::KeyPoint, cv::KeyPoint>> kpt_pairs_copy = kpt_pairs;
  kpt_pairs.clear();
  for (int i = 0; i < mask.total(); ++i) {
    if (mask.at<uchar>(i, 0)) {
      kpt_pairs.push_back(kpt_pairs_copy[i]);
    }
  }

  SE3 kf_pose = Converter::toSE3Quat(pKFi->GetPoseInverse()), f_pose = Converter::toSE3Quat(pF->mTcw_pnp.inv());
  if (R.inverse().allFinite()) {
    SE3 pose_kf_f = Converter::toSE3Quat(pKFi->GetPose() * pF->GetPoseInverse());
    LOG(INFO) << std::endl << pose_kf_f.rotationMatrix() << std::endl << R.inverse();
    Eigen::Quaterniond q_from_kf(kf_pose.rotationMatrix() * R.inverse());
    Eigen::Quaterniond q_from_kf_wrong(kf_pose.rotationMatrix() * R);
    Eigen::Quaterniond q_f(f_pose.unit_quaternion());
    Eigen::Quaterniond q_kf_f(pose_kf_f.rotationMatrix()), q_I = Eigen::Quaterniond::Identity();
    LOG(INFO) << "from kf:" << std::endl << q_from_kf.toRotationMatrix() << std::endl
              << "f: " << std::endl << q_f.toRotationMatrix();
    LOG(INFO) << fmt::format("diff {:.1f} degree, diff R {:.1f} degree, diff wrong {:.1f} degree",
                             q_from_kf.angularDistance(q_f) / M_PI * 180,
                             q_I.angularDistance(q_kf_f) / M_PI * 180,
                             q_from_kf_wrong.angularDistance(q_f) / M_PI * 180);
  }
  return true;
}

bool DescMap::ValidateByRenderedDistMap(cv::Mat &dist_img_f) {
  std::vector<cv::Point2f> dist_points;
  for (int y = 0; y < hG[0]; ++y) {
    for (int x = 0; x < wG[0]; ++x) {
      float dist = dist_img_f.at<float>(cv::Point(x, y));
      if (dist > 0) {
        dist_points.emplace_back(x, y);
      }
    }
  }
  const cv::Mat Rcw_pnp = mpCurrentFrame->mTcw_pnp.rowRange(0, 3).colRange(0, 3);
  const cv::Mat tcw_pnp = mpCurrentFrame->mTcw_pnp.rowRange(0, 3).col(3);

  typedef nanoflann::KDTreeSingleIndexAdaptor<
      nanoflann::L2_Simple_Adaptor<float, FLANNCvPoint2f>, FLANNCvPoint2f, 2>
      KDTreeCvPoint2f;
  FLANNCvPoint2f pc = FLANNCvPoint2f(dist_points.size(), dist_points.data());
  KDTreeCvPoint2f kdtree = KDTreeCvPoint2f(2, pc, nanoflann::KDTreeSingleIndexAdaptorParams(10));
  kdtree.buildIndex();

  double sum_dist = 0, sum_f = 0;
  Frame *pKF = all_keyframes[mnMatchKeyFrameDBId].get();
  for (int f_id = 0; f_id < pKF->N; ++f_id) {
    MapPoint *pMP = pKF->GetMapPoint(f_id);
    if (pMP && !pMP->isBad() && !pMP->mbEnhanced) {
      cv::Mat p3Dw = pMP->GetWorldPos();
      cv::Mat p3Dc = Rcw_pnp * p3Dw + tcw_pnp;

      if (p3Dc.at<float>(2) < 0.0f) continue;

      const float invz = 1 / p3Dc.at<float>(2);
      const float x = p3Dc.at<float>(0) * invz;
      const float y = p3Dc.at<float>(1) * invz;

      const float u = mpCurrentFrame->fx * x + mpCurrentFrame->cx;
      const float v = mpCurrentFrame->fy * y + mpCurrentFrame->cy;

      // Point must be inside the image
      if (!mpCurrentFrame->IsInImage(u, v)) continue;

      cv::Point2f ptF(u, v);
      float p_query[2] = {ptF.x, ptF.y};
      nanoflann::KNNResultSet<float, int, int> result_set(1);
      int ret_index[1];
      float ret_dist[1];
      result_set.init(ret_index, ret_dist);

      kdtree.findNeighbors(result_set, p_query, nanoflann::SearchParams());
      cv::Point2f npt = dist_points[ret_index[0]];
      double pix_dist = cv::norm(npt - ptF);
      if (pix_dist < 2) {
        float dist_proj = cv::norm(p3Dc);
        float dist_render = dist_img_f.at<float>(cv::Point(npt.x + 0.5, npt.y + 0.5));
        sum_f += 1;
        sum_dist += fabs(dist_proj - dist_render);
      }
    }
  }
  double mean_dist = sum_dist / sum_f;
  LOG(INFO) << "sum_f: " << int(sum_f) << " mean dist diff: " << mean_dist;
  if (mean_dist > 0.1) {
    LOG(WARNING) << ">>>>>>> dist not match <<<<<<<";
    return false;
  }
  return true;
}

void DescMap::SearchLocalPoints() {
  int nmatches = 0;

  int nToMatch = 0;
  // Project points in frame and check its visibility
  for (std::vector<MapPoint *>::iterator vit = mvpLocalMapPoints.begin(), vend = mvpLocalMapPoints.end(); vit != vend;
       vit++) {
    MapPoint *pMP = *vit;
    // if (pMP->mnLastFrameSeen == mpCurrentFrame->mnId)
    //   continue;
    if (pMP->isBad())
      continue;
    // Project (this fills MapPoint variables for matching)
    if (mpCurrentFrame->IsInFrustum(pMP, 0.5)) {
      pMP->IncreaseVisible();
      nToMatch++;
    }
  }

  if (nToMatch > 0) {
    std::unique_ptr<ORBmatcher> matcher = std::make_unique<ORBmatcher>(0.8, false);
    if (use_superpoint)
      matcher = std::make_unique<SPMatcher>(1.0);
    int th = 1;

    // If the camera has been relocalised recently, perform a coarser search
    // if (mpCurrentFrame->mnId < mnLastRelocFrameId + 2)
    //   th = 5;
    nmatches = matcher->SearchByProjection(*mpCurrentFrame, mvpLocalMapPoints, th);
  }

  LOG(INFO) << fmt::format(fmt::fg(fmt::color::brown),
                           "mvpLocalMapPoints.size(): {}, nToMatch: {}, {} nmatches @ {}",
                           mvpLocalMapPoints.size(), nToMatch, nmatches, __func__);
}

}