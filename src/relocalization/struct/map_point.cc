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

#include "relocalization/feature_matcher.h"
#include "relocalization/sp_matcher.h"
#include "relocalization/desc_map.h"
#include "relocalization/struct/map_point.h"
#include "relocalization/struct/frame.h"
#include "util/global_calib.h"

namespace dsl::relocalization {

long unsigned int MapPoint::nNextId = 1;

MapPoint::MapPoint() {}

/**
 * @brief
 * @param Pos global pos
 * @param pFrame ptr to the frame create the MapPoint
 * @param idxF idx in the mDescriptors of pFrame
 * @param _index_global index in the global map
 */
MapPoint::MapPoint(const cv::Mat &Pos, DescMap *pMap, Frame *pFrame, const int idxF, const int _index_global) :
    idx_in_surfel_map(_index_global), mnVisible(1), mnFound(1), mpMap(pMap), mnFirstKFid(pFrame->mnId) {
  Pos.copyTo(mWorldPos);
  cv::Mat Ow = pFrame->GetCameraCenter();
  mNormalVector = mWorldPos - Ow;
  mNormalVector = mNormalVector / cv::norm(mNormalVector);

  cv::Mat PC = Pos - Ow;
  const float dist = cv::norm(PC);
  const int level = pFrame->mvKeysUn[idxF].octave;
  const float levelScaleFactor = pFrame->mvScaleFactors[level];
  const int nLevels = pFrame->mnScaleLevels;

  mfMaxDistance = dist * levelScaleFactor;
  mfMinDistance = mfMaxDistance / pFrame->mvScaleFactors[nLevels - 1];

  pFrame->mDescriptors.row(idxF).copyTo(mDescriptor);

  mnId = nNextId++;
}

bool MapPoint::isBad() {
  return mbBad;
}

/**
 * @brief will erase from mpMap totally
 * not removed from recent map points, need to do it manually!
 */
void MapPoint::SetBadFlag() {
  std::map<Frame *, size_t> obs;
  {
    mbBad = true;
    obs = mObservations;
    mObservations.clear();
  }
  for (std::map<Frame *, size_t>::iterator mit = obs.begin(), mend = obs.end(); mit != mend; mit++) {
    Frame *pKF = mit->first;
    pKF->EraseMapPointMatch(mit->second);
  }
  mpMap->EraseMapPoint(this);
  // no EraseFromRecentMapPoints, since we need lit for iteration
}

cv::Mat MapPoint::GetWorldPos() {
  return mWorldPos.clone();
}

float MapPoint::GetMinDistanceInvariance() {
  return 0.8f * mfMinDistance;
}

float MapPoint::GetMaxDistanceInvariance() {
  return 1.2f * mfMaxDistance;
}

cv::Mat MapPoint::GetNormal() {
  return mNormalVector.clone();
}

int MapPoint::PredictScale(const float currentDist, Frame *pF) {
  float ratio = mfMaxDistance / currentDist;

  int nScale = ceil(log(ratio) / pF->mfLogScaleFactor);
  if (nScale < 0)
    nScale = 0;
  else if (nScale >= pF->mnScaleLevels)
    nScale = pF->mnScaleLevels - 1;

  return nScale;
}

void MapPoint::IncreaseVisible(int n) {
  mnVisible += n;
}

void MapPoint::IncreaseFound(int n) {
  mnFound += n;
}

float MapPoint::GetFoundRatio() {
  return static_cast<float>(mnFound) / mnVisible;
}

void MapPoint::AddObservation(Frame *pF, size_t idx) {
  if (mObservations.count(pF))
    return;
  mObservations[pF] = idx;
}

void MapPoint::EraseObservation(Frame *pF) {
  bool bBad = false;
  {
    if (mObservations.count(pF)) {
      int idx = mObservations[pF];
      // nObs--;

      mObservations.erase(pF);

      // if (mpRefKF == pKF)
      //   mpRefKF = mObservations.begin()->first;

      // If only 2 observations or less, discard point
      if (mObservations.size() <= 2)
        bBad = true;
    }
  }

  if (bBad) {
    SetBadFlag();
    mpMap->EraseFromRecentMapPoints(this);
  }
}

std::map<Frame *, size_t> &MapPoint::GetObservations() {
  return mObservations;
}

bool MapPoint::IsInKeyFrame(Frame *pF) {
  return (mObservations.count(pF));
}

void MapPoint::Replace(MapPoint *pMP) {
  if (pMP->mnId == this->mnId)
    return;
  int nvisible, nfound;
  std::map<Frame *, size_t> obs;

  {
    obs = mObservations;
    mObservations.clear();
    mbBad = true;
    nvisible = mnVisible;
    nfound = mnFound;
    mpReplaced = pMP;
  }

  for (std::map<Frame *, size_t>::iterator mit = obs.begin(), mend = obs.end(); mit != mend; mit++) {
    // Replace measurement in keyframe
    Frame *pKF = mit->first;

    if (!pMP->IsInKeyFrame(pKF)) {
      pKF->ReplaceMapPointMatch(mit->second, pMP);
      pMP->AddObservation(pKF, mit->second);
    } else {
      pKF->EraseMapPointMatch(mit->second);
      // WARNING: OOB, nfound & nvisbile need to be decreased?
    }
  }
  pMP->IncreaseFound(nfound);
  pMP->IncreaseVisible(nvisible);
  pMP->ComputeDistinctiveDescriptors();

  // TODO: delete this map point in the desc_map
  mpMap->EraseMapPoint(this);
  mpMap->EraseFromRecentMapPoints(this);
}

void MapPoint::UpdateNormalAndDepth() {
  std::map<Frame *, size_t> observations;
  Frame *pRefKF;
  cv::Mat Pos;
  {
    if (mbBad)
      return;
    observations = mObservations;
    pRefKF = observations.begin()->first; // mpRefKF;
    Pos = mWorldPos.clone();
  }

  if (observations.empty())
    return;

  cv::Mat normal = cv::Mat::zeros(3, 1, CV_32F);
  int n = 0;
  for (std::map<Frame *, size_t>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++) {
    Frame *pKF = mit->first;
    cv::Mat Owi = pKF->GetCameraCenter();
    cv::Mat normali = mWorldPos - Owi;
    normal = normal + normali / cv::norm(normali);
    n++;
  }

  cv::Mat PC = Pos - pRefKF->GetCameraCenter();
  const float dist = cv::norm(PC);
  const int level = pRefKF->mvKeysUn[observations[pRefKF]].octave;
  const float levelScaleFactor = pRefKF->mvScaleFactors[level];
  const int nLevels = pRefKF->mnScaleLevels;

  {
    if (!use_superpoint) {
      mfMaxDistance = dist * levelScaleFactor;
      mfMinDistance = mfMaxDistance / pRefKF->mvScaleFactors[nLevels - 1];
    }
    mNormalVector = normal / n;
  }
}

void MapPoint::ComputeDistinctiveDescriptors() {
  // Retrieve all observed descriptors
  std::vector<cv::Mat> vDescriptors;

  std::map<Frame *, size_t> observations;

  if (mbBad)
    return;
  observations = mObservations;

  if (observations.empty())
    return;

  vDescriptors.reserve(observations.size());

  for (std::map<Frame *, size_t>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++) {
    Frame *pKF = mit->first;

    // if (!pKF->isBad())
    vDescriptors.push_back(pKF->mDescriptors.row(mit->second));
  }

  if (vDescriptors.empty())
    return;

  // Compute distances between them
  const size_t N = vDescriptors.size();

  float Distances[N][N];
  for (size_t i = 0; i < N; i++) {
    Distances[i][i] = 0;
    for (size_t j = i + 1; j < N; j++) {
      float distij;
      if (!use_superpoint)
        distij = ORBmatcher::DescriptorDistance(vDescriptors[i], vDescriptors[j]);
      else
        distij = SPMatcher::DescriptorDistance(vDescriptors[i], vDescriptors[j]);
      Distances[i][j] = distij;
      Distances[j][i] = distij;
    }
  }

  // Take the descriptor with least median distance to the rest
  float BestMedian = std::numeric_limits<float>::max();
  int BestIdx = 0;
  for (size_t i = 0; i < N; i++) {
    std::vector<float> vDists(Distances[i], Distances[i] + N);
    std::sort(vDists.begin(), vDists.end());
    float median = vDists[0.5 * (N - 1)];

    if (median < BestMedian) {
      BestMedian = median;
      BestIdx = i;
    }
  }

  mDescriptor = vDescriptors[BestIdx].clone();
}

unsigned long MapPoint::GetId() {
  return mnId;
}

// mObservations to mFrameIdFeatId
void MapPoint::PrepareForSaving() {
  mFrameIdFeatId.clear();
  for (auto &&observation: mObservations) {
    mFrameIdFeatId[observation.first->mnId] = observation.second;
  }
}

// mFrameIdFeatId to mObservations
void MapPoint::Restore(const std::map<unsigned long, std::unique_ptr<Frame> > &all_keyframes) {
  mObservations.clear();
  for (auto &&it: mFrameIdFeatId) {
    unsigned long frame_id = it.first;
    size_t feat_id = it.second;
    AddObservation(all_keyframes.at(frame_id).get(), feat_id);
  }
}

template<class Archive>
void MapPoint::serialize(Archive &ar, const unsigned int version) {
  ar & mnId & nNextId;
  // ar & mConnectedMapPointWeights;
  ar & neighbor_points_global_ids;
  // Tracking related vars
  ar & mbBad;
  ar & idx_in_surfel_map;
  ar & mWorldPos;
  ar & mDescriptor;
  ar & mNormalVector;
  ar & mnVisible & mnFound;
  ar & mfMinDistance & mfMaxDistance;
  ar & mbOnPlane;
  ar & mFrameIdFeatId;
}
template void MapPoint::serialize(boost::archive::binary_iarchive &, const unsigned int);
template void MapPoint::serialize(boost::archive::binary_oarchive &, const unsigned int);

} // namespace dsl::relocalization