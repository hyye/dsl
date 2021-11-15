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

#include "relocalization/sp_matcher.h"
#include "util/global_calib.h"
#include <opencv2/cudafeatures2d.hpp>

namespace dsl::relocalization {

const float SPMatcher::TH_HIGH = 0.7;
const float SPMatcher::TH_LOW = 0.3;
const int SPMatcher::HISTO_LENGTH = 30;

SPMatcher::SPMatcher(float nnratio) : ORBmatcher(nnratio, false) {}

float SPMatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b) {
  float dist = (float) cv::norm(a, b, cv::NORM_L2);
  return dist;
}

int SPMatcher::SearchByProjection(Frame &F, const std::vector<MapPoint *> &vpMapPoints, const float th) {
  int nmatches = 0;

  const bool bFactor = th != 1.0;

  for (size_t iMP = 0; iMP < vpMapPoints.size(); iMP++) {
    MapPoint *pMP = vpMapPoints[iMP];
    if (!pMP->mbTrackInView)
      continue;

    if (pMP->isBad())
      continue;

    const int &nPredictedLevel = pMP->mnTrackScaleLevel;

    // The size of the window will depend on the viewing direction
    // by mpCurrentFrame->IsInFrustum
    float r = RadiusByViewingCos(pMP->mTrackViewCos);

    if (bFactor)
      r *= th;

    const std::vector<size_t> vIndices =
        F.GetFeaturesInArea(pMP->mTrackProjX,
                            pMP->mTrackProjY,
                            r * F.mvScaleFactors[nPredictedLevel],
                            nPredictedLevel - 1,
                            nPredictedLevel);

    if (vIndices.empty())
      continue;

    const cv::Mat MPdescriptor = pMP->GetDescriptor();

    float bestDist = std::numeric_limits<float>::max();
    int bestLevel = -1;
    float bestDist2 = std::numeric_limits<float>::max();
    int bestLevel2 = -1;
    int bestIdx = -1;

    // Get best and second matches with near keypoints
    for (std::vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++) {
      const size_t idx = *vit;

      /// We do not erase obseravtions, i.e. set mbBad flag
      /// and not have SearchByProjection by frames
      // if (F.mvpMapPoints[idx])
      //   if (F.mvpMapPoints[idx]->Observations() > 0)
      //     continue;

      const cv::Mat &d = F.mDescriptors.row(idx);

      const float dist = SPMatcher::DescriptorDistance(MPdescriptor, d);

      if (dist < bestDist) {
        bestDist2 = bestDist;
        bestDist = dist;
        bestLevel2 = bestLevel;
        bestLevel = F.mvKeysUn[idx].octave;
        bestIdx = idx;
      } else if (dist < bestDist2) {
        bestLevel2 = F.mvKeysUn[idx].octave;
        bestDist2 = dist;
      }
    }

    // Apply ratio to second match (only if best and second are in the same scale level)
    if (bestDist <= TH_HIGH) {
      if (bestLevel == bestLevel2 && bestDist > mfNNratio * bestDist2)
        continue;

      if (!F.mvpMapPoints[bestIdx]) {
        F.mvpMapPoints[bestIdx] = pMP;
        nmatches++;
      }
    }
  }
  return nmatches;
}

int SPMatcher::Fuse(Frame *pF, const std::vector<MapPoint *> &vpMapPoints, const float th) {
  LOG(INFO) << "sp fuse???";
  cv::Mat Rcw = pF->GetRotation();
  cv::Mat tcw = pF->GetTranslation();

  const float &fx = pF->fx;
  const float &fy = pF->fy;
  const float &cx = pF->cx;
  const float &cy = pF->cy;

  cv::Mat Ow = pF->GetCameraCenter();

  int nFused = 0;

  const int nMPs = vpMapPoints.size();

  for (int i = 0; i < nMPs; i++) {
    MapPoint *pMP = vpMapPoints[i];

    if (!pMP)
      continue;

    if (pMP->isBad() || pMP->IsInKeyFrame(pF))
      continue;

    cv::Mat p3Dw = pMP->GetWorldPos();
    cv::Mat p3Dc = Rcw * p3Dw + tcw;

    // Depth must be positive
    if (p3Dc.at<float>(2) < 0.0f)
      continue;

    const float invz = 1 / p3Dc.at<float>(2);
    const float x = p3Dc.at<float>(0) * invz;
    const float y = p3Dc.at<float>(1) * invz;

    const float u = fx * x + cx;
    const float v = fy * y + cy;

    // Point must be inside the image
    if (!pF->IsInImage(u, v))
      continue;

    cv::Mat PO = p3Dw - Ow;
    const float dist3D = cv::norm(PO);

    if (!use_superpoint) {
      const float maxDistance = pMP->GetMaxDistanceInvariance();
      const float minDistance = pMP->GetMinDistanceInvariance();

      // Depth must be inside the scale pyramid of the image
      if (dist3D < minDistance || dist3D > maxDistance)
        continue;
    }

    // Viewing angle must be less than 60 deg
    cv::Mat Pn = pMP->GetNormal();

    if (PO.dot(Pn) < 0.5 * dist3D)
      continue;

    int nPredictedLevel = pMP->PredictScale(dist3D, pF);

    // Search in a radius
    const float radius = th * pF->mvScaleFactors[nPredictedLevel];

    const std::vector<size_t> vIndices = pF->GetFeaturesInArea(u, v, radius);

    if (vIndices.empty())
      continue;

    // Match to the most similar keypoint in the radius

    const cv::Mat dMP = pMP->GetDescriptor();

    float bestDist = std::numeric_limits<float>::max();
    int bestIdx = -1;
    for (std::vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++) {
      const size_t idx = *vit;

      const cv::KeyPoint &kp = pF->mvKeysUn[idx];

      const int &kpLevel = kp.octave;

      if (!use_superpoint) {
        if (kpLevel < nPredictedLevel - 1 || kpLevel > nPredictedLevel)
          continue;
      }

      const float &kpx = kp.pt.x;
      const float &kpy = kp.pt.y;
      const float ex = u - kpx;
      const float ey = v - kpy;
      const float e2 = ex * ex + ey * ey;

      if (e2 * pF->mvInvLevelSigma2[kpLevel] > 5.99 * 2)
        continue;

      const cv::Mat &dKF = pF->mDescriptors.row(idx);

      const float dist = SPMatcher::DescriptorDistance(dMP, dKF);

      if (dist < bestDist) {
        bestDist = dist;
        bestIdx = idx;
      }
    }

    // If there is already a MapPoint replace otherwise add new measurement
    // FIXME: TH_HIGH->TH_LOW
    if (bestDist <= TH_HIGH) {
      MapPoint *pMPinKF = pF->GetMapPoint(bestIdx);
      if (pMPinKF) {
        if (!pMPinKF->isBad()) {
          if (pMPinKF->GetObservations().size() > pMP->GetObservations().size())
            pMP->Replace(pMPinKF);
          else
            pMPinKF->Replace(pMP);
        }
      } else {
        pMP->AddObservation(pF, bestIdx);
        pF->AddMapPoint(pMP, bestIdx);
      }
      nFused++;
    }
  }

  return nFused;
}

/// \brief For modify the vpMapPoints
/// \param pF
/// \param vpMapPoints
/// \param th
/// \return
int SPMatcher::FuseNew(Frame *pF, std::vector<MapPoint *> &vpMapPoints, const float th) {
  LOG(INFO) << "sp fuse new???";
  cv::Mat Rcw = pF->GetRotation();
  cv::Mat tcw = pF->GetTranslation();

  const float &fx = pF->fx;
  const float &fy = pF->fy;
  const float &cx = pF->cx;
  const float &cy = pF->cy;

  cv::Mat Ow = pF->GetCameraCenter();

  int nFused = 0;

  const int nMPs = vpMapPoints.size();

  for (int i = 0; i < nMPs; i++) {
    MapPoint *pMP = vpMapPoints[i];

    if (!pMP)
      continue;

    if (pMP->isBad() || pMP->IsInKeyFrame(pF))
      continue;

    cv::Mat p3Dw = pMP->GetWorldPos();
    cv::Mat p3Dc = Rcw * p3Dw + tcw;

    // Depth must be positive
    if (p3Dc.at<float>(2) < 0.0f)
      continue;

    const float invz = 1 / p3Dc.at<float>(2);
    const float x = p3Dc.at<float>(0) * invz;
    const float y = p3Dc.at<float>(1) * invz;

    const float u = fx * x + cx;
    const float v = fy * y + cy;

    // Point must be inside the image
    if (!pF->IsInImage(u, v))
      continue;

    cv::Mat PO = p3Dw - Ow;
    const float dist3D = cv::norm(PO);

    if (!use_superpoint) {
      const float maxDistance = pMP->GetMaxDistanceInvariance();
      const float minDistance = pMP->GetMinDistanceInvariance();

      // Depth must be inside the scale pyramid of the image
      if (dist3D < minDistance || dist3D > maxDistance)
        continue;
    }

    // Viewing angle must be less than 60 deg
    cv::Mat Pn = pMP->GetNormal();

    if (PO.dot(Pn) < 0.5 * dist3D)
      continue;

    int nPredictedLevel = pMP->PredictScale(dist3D, pF);

    // Search in a radius
    const float radius = th * pF->mvScaleFactors[nPredictedLevel];

    const std::vector<size_t> vIndices = pF->GetFeaturesInArea(u, v, radius);

    if (vIndices.empty())
      continue;

    // Match to the most similar keypoint in the radius

    const cv::Mat dMP = pMP->GetDescriptor();

    float bestDist = std::numeric_limits<float>::max();
    int bestIdx = -1;
    for (std::vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++) {
      const size_t idx = *vit;

      const cv::KeyPoint &kp = pF->mvKeysUn[idx];

      const int &kpLevel = kp.octave;

      if (!use_superpoint) {
        if (kpLevel < nPredictedLevel - 1 || kpLevel > nPredictedLevel)
          continue;
      }

      const float &kpx = kp.pt.x;
      const float &kpy = kp.pt.y;
      const float ex = u - kpx;
      const float ey = v - kpy;
      const float e2 = ex * ex + ey * ey;

      if (e2 * pF->mvInvLevelSigma2[kpLevel] > 5.99 * 2)
        continue;

      const cv::Mat &dKF = pF->mDescriptors.row(idx);

      const float dist = SPMatcher::DescriptorDistance(dMP, dKF);

      if (dist < bestDist) {
        bestDist = dist;
        bestIdx = idx;
      }
    }

    // If there is already a MapPoint replace otherwise add new measurement
    // FIXME: TH_HIGH->TH_LOW
    if (bestDist <= TH_HIGH) {
      MapPoint *pMPinKF = pF->GetMapPoint(bestIdx);
      if (pMPinKF) {
        if (!pMPinKF->isBad()) {
          if (pMPinKF->GetObservations().size() > pMP->GetObservations().size())
            pMP->Replace(pMPinKF);
          else
            pMPinKF->Replace(pMP);
        }
      } else {
        pMP->AddObservation(pF, bestIdx);
        pF->AddMapPoint(pMP, bestIdx);
      }
      vpMapPoints[i] = static_cast<MapPoint *>(NULL);
      nFused++;
    }
  }

  return nFused;
}

// TODO:
int SPMatcher::SearchByBruteForce(Frame *pKF,
                                  Frame &F,
                                  std::vector<MapPoint *> &vpMapPointMatches,
                                  bool useEnhancedMPs) {
  const std::vector<MapPoint *> vpMapPointsKF = pKF->mvpMapPoints;

  vpMapPointMatches = std::vector<MapPoint *>(F.N, static_cast<MapPoint *>(NULL));

  int nmatches = 0;

  std::vector<int> fid_kfs;
  cv::Mat descriptors_kf, descriptors_f = F.mDescriptors;
  for (int fid_kf = 0; fid_kf < vpMapPointsKF.size(); ++fid_kf) {
    MapPoint *pMP = vpMapPointsKF[fid_kf];
    if (pMP && !pMP->isBad()) {
      if (!useEnhancedMPs && pMP->mbEnhanced) continue;
      // Note: seems slightly better
      // descriptors_kf.push_back(pKF->mDescriptors.row(fid_kf));

      // Note: Used in testing
      descriptors_kf.push_back(pMP->GetDescriptor());
      fid_kfs.push_back(fid_kf);

      // NOTE: Best, slightly slower
      // if (pMP->GetObservations().empty()) continue;
      //
      // for (auto &&it:pMP->GetObservations()) {
      //   descriptors_kf.push_back(it.first->mDescriptors.row(it.second));
      //   fid_kfs.push_back(fid_kf);
      // }
    }
  }

  // FIXME:!!!
  // cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
  cv::Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher();
  std::vector<std::vector<cv::DMatch> > knn_matches;
  cv::cuda::GpuMat desc_kf_gpu, desc_f_gpu;
  desc_kf_gpu.upload(descriptors_kf);
  desc_f_gpu.upload(descriptors_f);
  matcher->knnMatch(desc_kf_gpu, desc_f_gpu, knn_matches, 2);
  // matcher->knnMatch(descriptors_kf, descriptors_f, knn_matches, 2); // replace with keyframe's flann

  //-- Filter matches using the Lowe's ratio test
  std::vector<cv::DMatch> good_matches;
  for (size_t i = 0; i < knn_matches.size(); i++) {
    // FIXME: TH_LOW -> TH_HIGH
    if (knn_matches[i][0].distance < TH_HIGH && knn_matches[i][0].distance <= mfNNratio * knn_matches[i][1].distance) {
      good_matches.push_back(knn_matches[i][0]);
    }
  }
  // nmatches = good_matches.size();
  std::set<int> fid_kf_set, fid_f_set;
  for (size_t i = 0; i < good_matches.size(); i++) {
    cv::DMatch match = good_matches[i];
    int idx_fid_kf = match.queryIdx;
    int idx_fid_f = match.trainIdx;
    int fid_kf = fid_kfs[idx_fid_kf];
    int fid_f = idx_fid_f;
    if (!fid_kf_set.count(fid_kf) && !fid_f_set.count(fid_f)) {
      vpMapPointMatches[fid_f] = vpMapPointsKF[fid_kf];
      fid_kf_set.insert(fid_kf);
      fid_f_set.insert(fid_f);
      nmatches += 1;
    }
  }

  float min_dist = std::numeric_limits<float>::max(), max_dist = 0;
  for (size_t i = 0; i < knn_matches.size(); i++) {
    if (min_dist > knn_matches[i][0].distance) min_dist = knn_matches[i][0].distance;
    if (max_dist < knn_matches[i][0].distance) max_dist = knn_matches[i][0].distance;
  }
  LOG(WARNING) << "sp matcher bf pf,f,vpmp,b: " << nmatches << " " << min_dist << " " << max_dist;

  return nmatches;
}

// TODO:
int SPMatcher::SearchByBruteForce(const std::vector<MapPoint *> &vpMPs, Frame &F,
                                  std::vector<MapPoint *> &vpMapPointMatches, bool useEnhancedMPs) {
  vpMapPointMatches = std::vector<MapPoint *>(F.N, static_cast<MapPoint *>(NULL));

  int nmatches = 0;

  std::vector<int> pids;
  cv::Mat descriptors_kf, descriptors_f = F.mDescriptors;
  for (int pid = 0; pid < vpMPs.size(); ++pid) {
    MapPoint *pMP = vpMPs[pid];
    if (pMP && !pMP->isBad()) {
      if (!useEnhancedMPs && pMP->mbEnhanced) continue;
      descriptors_kf.push_back(pMP->GetDescriptor());
      pids.push_back(pid);
    }
  }

  // FIXME:!!!
  cv::Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher();
  std::vector<std::vector<cv::DMatch> > knn_matches;
  cv::cuda::GpuMat desc_kf_gpu, desc_f_gpu;
  desc_kf_gpu.upload(descriptors_kf);
  desc_f_gpu.upload(descriptors_f);
  matcher->knnMatch(desc_kf_gpu, desc_f_gpu, knn_matches, 2);
  // matcher->knnMatch(descriptors_kf, descriptors_f, knn_matches, 2); // replace with keyframe's flann

  //-- Filter matches using the Lowe's ratio test
  std::vector<cv::DMatch> good_matches;
  for (size_t i = 0; i < knn_matches.size(); i++) {
    // FIXME: to TH_LOW->TH_HIGH
    if (knn_matches[i][0].distance < TH_HIGH && knn_matches[i][0].distance <= mfNNratio * knn_matches[i][1].distance) {
      good_matches.push_back(knn_matches[i][0]);
    }
  }
  // nmatches = good_matches.size();
  std::set<int> pid_set, fid_f_set;
  for (size_t i = 0; i < good_matches.size(); i++) {
    cv::DMatch match = good_matches[i];
    int idx_fid_kf = match.queryIdx;
    int idx_fid_f = match.trainIdx;
    int pid = pids[idx_fid_kf];
    int fid_f = idx_fid_f;
    if (!pid_set.count(pid) && !fid_f_set.count(fid_f)) {
      vpMapPointMatches[fid_f] = vpMPs[pid];
      pid_set.insert(pid);
      fid_f_set.insert(fid_f);
      nmatches += 1;
    }
  }
  return nmatches;
}

// TODO:
int SPMatcher::SearchByProjection(Frame &CurrentFrame,
                                  const std::vector<MapPoint *> &vpMapPoints,
                                  const std::set<MapPoint *> &sAlreadyFound,
                                  const float th,
                                  const float ORBdist) {
  int nmatches = 0;

  const cv::Mat Rcw = CurrentFrame.mTcw_pnp.rowRange(0, 3).colRange(0, 3);
  const cv::Mat tcw = CurrentFrame.mTcw_pnp.rowRange(0, 3).col(3);
  const cv::Mat Ow = -Rcw.t() * tcw;

  for (size_t i = 0, iend = vpMapPoints.size(); i < iend; i++) {
    MapPoint *pMP = vpMapPoints[i];

    if (pMP) {
      if (!pMP->isBad() && !sAlreadyFound.count(pMP)) {
        //Project
        cv::Mat x3Dw = pMP->GetWorldPos();
        cv::Mat x3Dc = Rcw * x3Dw + tcw;

        const float xc = x3Dc.at<float>(0);
        const float yc = x3Dc.at<float>(1);
        const float invzc = 1.0 / x3Dc.at<float>(2);

        const float u = CurrentFrame.fx * xc * invzc + CurrentFrame.cx;
        const float v = CurrentFrame.fy * yc * invzc + CurrentFrame.cy;

        if (u < CurrentFrame.mnMinX || u > CurrentFrame.mnMaxX)
          continue;
        if (v < CurrentFrame.mnMinY || v > CurrentFrame.mnMaxY)
          continue;

        // Compute predicted scale level
        cv::Mat PO = x3Dw - Ow;
        float dist3D = cv::norm(PO);


        // const float maxDistance = pMP->GetMaxDistanceInvariance();
        // const float minDistance = pMP->GetMinDistanceInvariance();

        // // Depth must be inside the scale pyramid of the image
        // if (dist3D < minDistance || dist3D > maxDistance)
        //   continue;

        // int nPredictedLevel = pMP->PredictScale(dist3D, &CurrentFrame);

        int nPredictedLevel = 0;

        // Search in a window
        const float radius = th * CurrentFrame.mvScaleFactors[nPredictedLevel];

        const std::vector<size_t>
            vIndices2 = CurrentFrame.GetFeaturesInArea(u, v, radius, nPredictedLevel - 1, nPredictedLevel + 1);

        if (vIndices2.empty())
          continue;

        const cv::Mat dMP = pMP->GetDescriptor();

        float bestDist = std::numeric_limits<float>::max();
        int bestIdx2 = -1;

        for (std::vector<size_t>::const_iterator vit = vIndices2.begin(); vit != vIndices2.end(); vit++) {
          const size_t i2 = *vit;
          if (CurrentFrame.mvpMapPoints[i2])
            continue;

          const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);

          const float dist = SPMatcher::DescriptorDistance(dMP, d);

          // LOG(INFO) << dist;

          if (dist < bestDist) {
            bestDist = dist;
            bestIdx2 = i2;
          }
        }

        if (bestDist <= ORBdist) {
          CurrentFrame.mvpMapPoints[bestIdx2] = pMP;
          nmatches++;
        }

      }
    }
  }

  return nmatches;
}

} // namespace