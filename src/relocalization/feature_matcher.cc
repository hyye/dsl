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

#include "relocalization/feature_matcher.h"
#include <opencv2/cudafeatures2d.hpp>

namespace dsl {

namespace relocalization {

const int ORBmatcher::TH_HIGH = 100;
const int ORBmatcher::TH_LOW = 50;
const int ORBmatcher::HISTO_LENGTH = 30;

ORBmatcher::ORBmatcher(float nnratio, bool checkOri) : mfNNratio(nnratio), mbCheckOrientation(checkOri) {
}

void ORBmatcher::ComputeThreeMaxima(std::vector<int> *histo, const int L, int &ind1, int &ind2, int &ind3) {
  int max1 = 0;
  int max2 = 0;
  int max3 = 0;

  for (int i = 0; i < L; i++) {
    const int s = histo[i].size();
    if (s > max1) {
      max3 = max2;
      max2 = max1;
      max1 = s;
      ind3 = ind2;
      ind2 = ind1;
      ind1 = i;
    } else if (s > max2) {
      max3 = max2;
      max2 = s;
      ind3 = ind2;
      ind2 = i;
    } else if (s > max3) {
      max3 = s;
      ind3 = i;
    }
  }

  if (max2 < 0.1f * (float) max1) {
    ind2 = -1;
    ind3 = -1;
  } else if (max3 < 0.1f * (float) max1) {
    ind3 = -1;
  }
}

// Bit set count operation from
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
int ORBmatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b) {
  const int *pa = a.ptr<int32_t>();
  const int *pb = b.ptr<int32_t>();

  int dist = 0;

  for (int i = 0; i < 8; i++, pa++, pb++) {
    unsigned int v = *pa ^*pb;
    v = v - ((v >> 1) & 0x55555555);
    v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
    dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
  }

  return dist;
}

float ORBmatcher::RadiusByViewingCos(const float viewCos) {
  if (viewCos > 0.998)
    return 2.5;
  else
    return 4.0;
}

int ORBmatcher::SearchByProjection(Frame &CurrentFrame,
                                   Frame *pKF,
                                   const std::set<MapPoint *> &sAlreadyFound,
                                   const float th,
                                   const float ORBdist) {
  int nmatches = 0;

  const cv::Mat Rcw = CurrentFrame.mTcw_pnp.rowRange(0, 3).colRange(0, 3);
  const cv::Mat tcw = CurrentFrame.mTcw_pnp.rowRange(0, 3).col(3);
  const cv::Mat Ow = -Rcw.t() * tcw;

  // Rotation Histogram (to check rotation consistency)
  std::vector<int> rotHist[HISTO_LENGTH];
  for (int i = 0; i < HISTO_LENGTH; i++)
    rotHist[i].reserve(500);
  const float factor = 1.0f / HISTO_LENGTH;

  // TODO: GetMapPointMatches
  const std::vector<MapPoint *> vpMPs = pKF->mvpMapPoints;

  for (size_t i = 0, iend = vpMPs.size(); i < iend; i++) {
    MapPoint *pMP = vpMPs[i];

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

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();

        // Depth must be inside the scale pyramid of the image
        if (dist3D < minDistance || dist3D > maxDistance)
          continue;

        int nPredictedLevel = pMP->PredictScale(dist3D, &CurrentFrame);

        // Search in a window
        const float radius = th * CurrentFrame.mvScaleFactors[nPredictedLevel];

        const std::vector<size_t>
            vIndices2 = CurrentFrame.GetFeaturesInArea(u, v, radius, nPredictedLevel - 1, nPredictedLevel + 1);

        if (vIndices2.empty())
          continue;

        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = 256;
        int bestIdx2 = -1;

        for (std::vector<size_t>::const_iterator vit = vIndices2.begin(); vit != vIndices2.end(); vit++) {
          const size_t i2 = *vit;
          if (CurrentFrame.mvpMapPoints[i2])
            continue;

          const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);

          const int dist = DescriptorDistance(dMP, d);

          if (dist < bestDist) {
            bestDist = dist;
            bestIdx2 = i2;
          }
        }

        if (bestDist <= ORBdist) {
          CurrentFrame.mvpMapPoints[bestIdx2] = pMP;
          nmatches++;

          if (mbCheckOrientation) {
            float rot = pKF->mvKeysUn[i].angle - CurrentFrame.mvKeysUn[bestIdx2].angle;
            if (rot < 0.0)
              rot += 360.0f;
            int bin = round(rot * factor);
            if (bin == HISTO_LENGTH)
              bin = 0;
            assert(bin >= 0 && bin < HISTO_LENGTH);
            rotHist[bin].push_back(bestIdx2);
          }
        }

      }
    }
  }

  if (mbCheckOrientation) {
    int ind1 = -1;
    int ind2 = -1;
    int ind3 = -1;

    ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

    for (int i = 0; i < HISTO_LENGTH; i++) {
      if (i != ind1 && i != ind2 && i != ind3) {
        for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++) {
          CurrentFrame.mvpMapPoints[rotHist[i][j]] = NULL;
          nmatches--;
        }
      }
    }
  }

  return nmatches;
}

/**
 * @brief
 * @param F
 * @param vpMapPoints
 * @param th
 * @return # of matches
 */
int ORBmatcher::SearchByProjection(Frame &F, const std::vector<MapPoint *> &vpMapPoints, const float th) {
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

    int bestDist = 256;
    int bestLevel = -1;
    int bestDist2 = 256;
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

      const int dist = DescriptorDistance(MPdescriptor, d);

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

int ORBmatcher::SearchByProjection(Frame &F,
                                   const std::vector<MapPoint *> &vpMapPoints,
                                   std::vector<MapPoint *> &vpOutMPs,
                                   const float th) {
  int nmatches = 0;

  vpOutMPs = std::vector<MapPoint *>(F.N, static_cast<MapPoint *>(NULL));

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

    int bestDist = 256;
    int bestLevel = -1;
    int bestDist2 = 256;
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

      const int dist = DescriptorDistance(MPdescriptor, d);

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

      if (!vpOutMPs[bestIdx]) {
        vpOutMPs[bestIdx] = pMP;
        nmatches++;
      }
    }
  }

  return nmatches;

}

int ORBmatcher::SearchByProjectionCustom(Frame &F,
                                   const std::vector<MapPoint *> &vpMapPoints,
                                   std::vector<MapPoint *> &vpOutMPs,
                                   const float th) {
  int nmatches = 0;

  vpOutMPs = std::vector<MapPoint *>(F.N, static_cast<MapPoint *>(NULL));

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

    int bestDist = 256;
    int bestLevel = -1;
    int bestDist2 = 256;
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

      const int dist = DescriptorDistance(MPdescriptor, d);

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
      // CUSTOM to check best only
      // if (bestLevel == bestLevel2 && bestDist > mfNNratio * bestDist2)
      //   continue;

      if (!vpOutMPs[bestIdx]) {
        vpOutMPs[bestIdx] = pMP;
        nmatches++;
      }
    }
  }

  return nmatches;

}

int ORBmatcher::SearchByProjection(Frame &CurrentFrame,
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

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();

        // Depth must be inside the scale pyramid of the image
        if (dist3D < minDistance || dist3D > maxDistance)
          continue;

        int nPredictedLevel = pMP->PredictScale(dist3D, &CurrentFrame);

        // Search in a window
        const float radius = th * CurrentFrame.mvScaleFactors[nPredictedLevel];

        const std::vector<size_t>
            vIndices2 = CurrentFrame.GetFeaturesInArea(u, v, radius, nPredictedLevel - 1, nPredictedLevel + 1);

        if (vIndices2.empty())
          continue;

        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = 256;
        int bestIdx2 = -1;

        for (std::vector<size_t>::const_iterator vit = vIndices2.begin(); vit != vIndices2.end(); vit++) {
          const size_t i2 = *vit;
          if (CurrentFrame.mvpMapPoints[i2])
            continue;

          const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);

          const int dist = DescriptorDistance(dMP, d);

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

bool ORBmatcher::SearchFromFeatVec(Frame &F,
                                   std::vector<unsigned int> &feat_ids,
                                   cv::Mat &query_descriptor,
                                   int &feat_id) {
  if (feat_ids.size() <= 2) return false;
  int bestDist1 = 256;
  int bestIdxF = -1;
  int bestDist2 = 256;
  const cv::Mat &dMF = query_descriptor;
  for (auto realIdxKF : feat_ids) {
    const cv::Mat &dF = F.mDescriptors.row(realIdxKF);

    const int dist = DescriptorDistance(dF, dMF);

    if (dist < bestDist1) {
      bestDist2 = bestDist1;
      bestDist1 = dist;
      bestIdxF = realIdxKF;
    } else if (dist < bestDist2) {
      bestDist2 = dist;
    }
  }
  if (bestDist1 <= TH_LOW) {
    if (static_cast<float>(bestDist1) < mfNNratio * static_cast<float>(bestDist2)) {
      feat_id = bestIdxF;
      return true;
    }
  }
  return false;
}

int ORBmatcher::SearchByBruteForce(const std::vector<MapPoint *> &vpMPs,
                                   Frame &F,
                                   std::vector<MapPoint *> &vpMapPointMatches,
                                   bool useEnhancedMPs) {
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
  // cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
  cv::Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
  std::vector<std::vector<cv::DMatch> > knn_matches;
  cv::cuda::GpuMat desc_kf_gpu, desc_f_gpu;
  desc_kf_gpu.upload(descriptors_kf);
  desc_f_gpu.upload(descriptors_f);
  matcher->knnMatch(desc_kf_gpu, desc_f_gpu, knn_matches, 2);
  // matcher->knnMatch(descriptors_kf, descriptors_f, knn_matches, 2); // replace with keyframe's flann

  //-- Filter matches using the Lowe's ratio test
  std::vector<cv::DMatch> good_matches;
  for (size_t i = 0; i < knn_matches.size(); i++) {
    if (knn_matches[i][0].distance < TH_LOW && knn_matches[i][0].distance < mfNNratio * knn_matches[i][1].distance) {
      good_matches.push_back(knn_matches[i][0]);
    }
  }
  nmatches = good_matches.size();
  for (size_t i = 0; i < good_matches.size(); i++) {
    cv::DMatch match = good_matches[i];
    int idx_fid_kf = match.queryIdx;
    int idx_fid_f = match.trainIdx;
    int pid = pids[idx_fid_kf];
    int fid_f = idx_fid_f;
    vpMapPointMatches[fid_f] = vpMPs[pid];
  }
  return nmatches;
}

int ORBmatcher::SearchByBruteForce(Frame *pKF,
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
      descriptors_kf.push_back(pKF->mDescriptors.row(fid_kf));
      fid_kfs.push_back(fid_kf);
    }
  }
  // cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
  cv::Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
  std::vector<std::vector<cv::DMatch> > knn_matches;
  cv::cuda::GpuMat desc_kf_gpu, desc_f_gpu;
  desc_kf_gpu.upload(descriptors_kf);
  desc_f_gpu.upload(descriptors_f);
  matcher->knnMatch(desc_kf_gpu, desc_f_gpu, knn_matches, 2);
  // matcher->knnMatch(descriptors_kf, descriptors_f, knn_matches, 2); // replace with keyframe's flann

  //-- Filter matches using the Lowe's ratio test
  std::vector<cv::DMatch> good_matches;
  for (size_t i = 0; i < knn_matches.size(); i++) {
    if (knn_matches[i][0].distance < TH_LOW && knn_matches[i][0].distance < mfNNratio * knn_matches[i][1].distance) {
      good_matches.push_back(knn_matches[i][0]);
    }
  }
  nmatches = good_matches.size();
  for (size_t i = 0; i < good_matches.size(); i++) {
    cv::DMatch match = good_matches[i];
    int idx_fid_kf = match.queryIdx;
    int idx_fid_f = match.trainIdx;
    int fid_kf = fid_kfs[idx_fid_kf];
    int fid_f = idx_fid_f;
    vpMapPointMatches[fid_f] = vpMapPointsKF[fid_kf];
  }
  return nmatches;
}

int ORBmatcher::SearchByBruteForce(Frame *pKF,
                                   Frame &F,
                                   std::vector<std::pair<int, int>> &corres_id,
                                   bool useEnhancedMPs) {

  const std::vector<MapPoint *> vpMapPointsKF = pKF->mvpMapPoints;

  int nmatches = 0;
  corres_id.clear();

  std::vector<int> fid_kfs;
  cv::Mat descriptors_kf, descriptors_f = F.mDescriptors;
  for (int fid_kf = 0; fid_kf < vpMapPointsKF.size(); ++fid_kf) {
    MapPoint *pMP = vpMapPointsKF[fid_kf];
    if (pMP && !pMP->isBad()) {
      if (!useEnhancedMPs && pMP->mbEnhanced) continue;
      descriptors_kf.push_back(pKF->mDescriptors.row(fid_kf));
      fid_kfs.push_back(fid_kf);
    }
  }

  // cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
  cv::Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
  std::vector<std::vector<cv::DMatch> > knn_matches;
  cv::cuda::GpuMat desc_kf_gpu, desc_f_gpu;
  desc_kf_gpu.upload(descriptors_kf);
  desc_f_gpu.upload(descriptors_f);
  matcher->knnMatch(desc_kf_gpu, desc_f_gpu, knn_matches, 2);
  // matcher->knnMatch(descriptors_kf, descriptors_f, knn_matches, 2); // replace with keyframe's flann

  //-- Filter matches using the Lowe's ratio test
  std::vector<cv::DMatch> good_matches;
  for (size_t i = 0; i < knn_matches.size(); i++) {
    if (knn_matches[i][0].distance < TH_LOW && knn_matches[i][0].distance < mfNNratio * knn_matches[i][1].distance) {
      good_matches.push_back(knn_matches[i][0]);
    }
  }
  nmatches = good_matches.size();
  for (size_t i = 0; i < good_matches.size(); i++) {
    cv::DMatch match = good_matches[i];
    int idx_fid_kf = match.queryIdx;
    int idx_fid_f = match.trainIdx;
    int fid_kf = fid_kfs[idx_fid_kf];
    int fid_f = idx_fid_f;
    corres_id.push_back(std::make_pair(fid_f, fid_kf));  // frame's, keyframe's
  }
  return nmatches;
}

int ORBmatcher::SearchByBoW(Frame *pKF, Frame &F, std::vector<MapPoint *> &vpMapPointMatches, bool useEnhancedMPs) {
  // TODO: GetMapPointMatches
  const std::vector<MapPoint *> vpMapPointsKF = pKF->mvpMapPoints;

  vpMapPointMatches = std::vector<MapPoint *>(F.N, static_cast<MapPoint *>(NULL));

  const DBoW2::FeatureVector &vFeatVecKF = pKF->mFeatVec;

  int nmatches = 0;

  std::vector<int> rotHist[HISTO_LENGTH];
  for (int i = 0; i < HISTO_LENGTH; i++)
    rotHist[i].reserve(500);
  const float factor = 1.0f / HISTO_LENGTH;

  // We perform the matching over ORB that belong to the same vocabulary node (at a certain level)
  DBoW2::FeatureVector::const_iterator KFit = vFeatVecKF.begin();
  DBoW2::FeatureVector::const_iterator Fit = F.mFeatVec.begin();
  DBoW2::FeatureVector::const_iterator KFend = vFeatVecKF.end();
  DBoW2::FeatureVector::const_iterator Fend = F.mFeatVec.end();

  while (KFit != KFend && Fit != Fend) {
    if (KFit->first == Fit->first) {
      const std::vector<unsigned int> vIndicesKF = KFit->second;
      const std::vector<unsigned int> vIndicesF = Fit->second;

      for (size_t iKF = 0; iKF < vIndicesKF.size(); iKF++) {
        const unsigned int realIdxKF = vIndicesKF[iKF];

        MapPoint *pMP = vpMapPointsKF[realIdxKF];

        if (!pMP)
          continue;

        if (pMP->isBad())
          continue;

        if (!useEnhancedMPs && pMP->mbEnhanced) continue;

        const cv::Mat &dKF = pKF->mDescriptors.row(realIdxKF);

        int bestDist1 = 256;
        int bestIdxF = -1;
        int bestDist2 = 256;

        for (size_t iF = 0; iF < vIndicesF.size(); iF++) {
          const unsigned int realIdxF = vIndicesF[iF];

          if (vpMapPointMatches[realIdxF])
            continue;

          const cv::Mat &dF = F.mDescriptors.row(realIdxF);

          const int dist = DescriptorDistance(dKF, dF);

          if (dist < bestDist1) {
            bestDist2 = bestDist1;
            bestDist1 = dist;
            bestIdxF = realIdxF;
          } else if (dist < bestDist2) {
            bestDist2 = dist;
          }
        }

        if (bestDist1 <= TH_LOW) {
          if (static_cast<float>(bestDist1) < mfNNratio * static_cast<float>(bestDist2)) {
            vpMapPointMatches[bestIdxF] = pMP;

            const cv::KeyPoint &kp = pKF->mvKeysUn[realIdxKF];

            if (mbCheckOrientation) {
              float rot = kp.angle - F.mvKeysUn[bestIdxF].angle;
              if (rot < 0.0)
                rot += 360.0f;
              int bin = round(rot * factor);
              if (bin == HISTO_LENGTH)
                bin = 0;
              assert(bin >= 0 && bin < HISTO_LENGTH);
              rotHist[bin].push_back(bestIdxF);
            }
            nmatches++;
          }
        }

      }

      KFit++;
      Fit++;
    } else if (KFit->first < Fit->first) {
      KFit = vFeatVecKF.lower_bound(Fit->first);
    } else {
      Fit = F.mFeatVec.lower_bound(KFit->first);
    }
  }

  if (mbCheckOrientation) {
    int ind1 = -1;
    int ind2 = -1;
    int ind3 = -1;

    ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

    for (int i = 0; i < HISTO_LENGTH; i++) {
      if (i == ind1 || i == ind2 || i == ind3)
        continue;
      for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++) {
        vpMapPointMatches[rotHist[i][j]] = static_cast<MapPoint *>(NULL);
        nmatches--;
      }
    }
  }

  return nmatches;
}

int ORBmatcher::SearchByBoW(DescMap *desc_map, Frame &F, std::vector<MapPoint *> &vpMapPointMatches) {

  const std::vector<MapPoint *> vpMapPointsKF = desc_map->GetMapPoints();

  vpMapPointMatches = std::vector<MapPoint *>(F.N, static_cast<MapPoint *>(NULL));

  const DBoW2::FeatureVector &vFeatVecKF = desc_map->mFeatVec;

  int nmatches = 0;

  std::vector<int> rotHist[HISTO_LENGTH];
  for (int i = 0; i < HISTO_LENGTH; i++)
    rotHist[i].reserve(500);
  const float factor = 1.0f / HISTO_LENGTH;

  // We perform the matching over ORB that belong to the same vocabulary node (at a certain level)
  DBoW2::FeatureVector::const_iterator KFit = vFeatVecKF.begin();
  DBoW2::FeatureVector::const_iterator Fit = F.mFeatVec.begin();
  DBoW2::FeatureVector::const_iterator KFend = vFeatVecKF.end();
  DBoW2::FeatureVector::const_iterator Fend = F.mFeatVec.end();

  while (KFit != KFend && Fit != Fend) {
    if (KFit->first == Fit->first) {
      const std::vector<unsigned int> vIndicesKF = KFit->second;
      const std::vector<unsigned int> vIndicesF = Fit->second;

      for (size_t iKF = 0; iKF < vIndicesKF.size(); iKF++) {
        const unsigned int realIdxKF = vIndicesKF[iKF];

        MapPoint *pMP = vpMapPointsKF[realIdxKF];

        if (!pMP)
          continue;

        if (pMP->isBad())
          continue;

        const cv::Mat &dKF = desc_map->mDescriptors.row(realIdxKF);

        int bestDist1 = 256;
        int bestIdxF = -1;
        int bestDist2 = 256;

        for (size_t iF = 0; iF < vIndicesF.size(); iF++) {
          const unsigned int realIdxF = vIndicesF[iF];

          if (vpMapPointMatches[realIdxF])
            continue;

          const cv::Mat &dF = F.mDescriptors.row(realIdxF);

          const int dist = DescriptorDistance(dKF, dF);

          if (dist < bestDist1) {
            bestDist2 = bestDist1;
            bestDist1 = dist;
            bestIdxF = realIdxF;
          } else if (dist < bestDist2) {
            bestDist2 = dist;
          }
        }

        if (bestDist1 <= TH_LOW) {
          if (static_cast<float>(bestDist1) < mfNNratio * static_cast<float>(bestDist2)) {
            vpMapPointMatches[bestIdxF] = pMP;
            // NOTE: no orientation
            nmatches++;
          }
        }

      }

      KFit++;
      Fit++;
    } else if (KFit->first < Fit->first) {
      KFit = vFeatVecKF.lower_bound(Fit->first);
    } else {
      Fit = F.mFeatVec.lower_bound(KFit->first);
    }
  }
  // NOTE: no orientation
  return nmatches;
}

int ORBmatcher::Fuse(Frame *pF, const std::vector<MapPoint *> &vpMapPoints, const float th) {
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

    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    cv::Mat PO = p3Dw - Ow;
    const float dist3D = cv::norm(PO);

    // Depth must be inside the scale pyramid of the image
    if (dist3D < minDistance || dist3D > maxDistance)
      continue;

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

    int bestDist = 256;
    int bestIdx = -1;
    for (std::vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++) {
      const size_t idx = *vit;

      const cv::KeyPoint &kp = pF->mvKeysUn[idx];

      const int &kpLevel = kp.octave;

      if (kpLevel < nPredictedLevel - 1 || kpLevel > nPredictedLevel)
        continue;

      const float &kpx = kp.pt.x;
      const float &kpy = kp.pt.y;
      const float ex = u - kpx;
      const float ey = v - kpy;
      const float e2 = ex * ex + ey * ey;

      if (e2 * pF->mvInvLevelSigma2[kpLevel] > 5.99)
        continue;

      const cv::Mat &dKF = pF->mDescriptors.row(idx);

      const int dist = DescriptorDistance(dMP, dKF);

      if (dist < bestDist) {
        bestDist = dist;
        bestIdx = idx;
      }
    }

    // If there is already a MapPoint replace otherwise add new measurement
    if (bestDist <= TH_LOW) {
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
int ORBmatcher::FuseNew(Frame *pF, std::vector<MapPoint *> &vpMapPoints, const float th) {
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

    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    cv::Mat PO = p3Dw - Ow;
    const float dist3D = cv::norm(PO);

    // Depth must be inside the scale pyramid of the image
    if (dist3D < minDistance || dist3D > maxDistance)
      continue;

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

    int bestDist = 256;
    int bestIdx = -1;
    for (std::vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++) {
      const size_t idx = *vit;

      const cv::KeyPoint &kp = pF->mvKeysUn[idx];

      const int &kpLevel = kp.octave;

      if (kpLevel < nPredictedLevel - 1 || kpLevel > nPredictedLevel)
        continue;

      const float &kpx = kp.pt.x;
      const float &kpy = kp.pt.y;
      const float ex = u - kpx;
      const float ey = v - kpy;
      const float e2 = ex * ex + ey * ey;

      if (e2 * pF->mvInvLevelSigma2[kpLevel] > 5.99)
        continue;

      const cv::Mat &dKF = pF->mDescriptors.row(idx);

      const int dist = DescriptorDistance(dMP, dKF);

      if (dist < bestDist) {
        bestDist = dist;
        bestIdx = idx;
      }
    }

    // If there is already a MapPoint replace otherwise add new measurement
    if (bestDist <= TH_LOW) {
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

} // namespace relocalization

} // namespace dsl