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

#include "relocalization/struct/map_point.h"
#include "relocalization/struct/frame.h"
#include "relocalization/desc_map.h"
#include "relocalization/converter.h"
#include "util/global_calib.h"

namespace dsl::relocalization {

long unsigned int Frame::nNextId = 1;
bool Frame::mbInitialComputations = true;
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

Frame::Frame(const cv::Mat &imageIn, const std::string timeStamp, ORBextractor *extractor, OrbVocabularyBinary *voc) :
    mpORBvocabulary(voc), mpORBextractor(extractor), mTimeStamp(timeStamp) {
  // Frame ID
  mnId = nNextId++;
  // Scale Level Info
  mnScaleLevels = mpORBextractor->GetLevels();
  mfScaleFactor = mpORBextractor->GetScaleFactor();
  mfLogScaleFactor = log(mfScaleFactor);
  mvScaleFactors = mpORBextractor->GetScaleFactors();
  mvInvScaleFactors = mpORBextractor->GetInverseScaleFactors();
  mvLevelSigma2 = mpORBextractor->GetScaleSigmaSquares();
  mvInvLevelSigma2 = mpORBextractor->GetInverseScaleSigmaSquares();

  // ORB extraction
  // std::thread threadExtractor(&Frame::ExtractORB, this, 0, imageIn);
  // threadExtractor.join();
  ExtractORB(0, imageIn);

  N = mvKeysUn.size();

  if (mvKeysUn.empty())
    return;

  mvpMapPoints = std::vector<MapPoint *>(N, static_cast<MapPoint *>(NULL));
  mvbOutlier = std::vector<bool>(N, false);

  // This is done only for the first Frame (or after a change in the calibration)
  if (mbInitialComputations) {
    mnMinX = 0.0f;
    mnMaxX = wG[0] - 1;
    mnMinY = 0.0f;
    mnMaxY = hG[0] - 1;

    mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) / (mnMaxX - mnMinX);
    mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / (mnMaxY - mnMinY);

    fx = fxG[0];
    fy = fyG[0];
    cx = cxG[0];
    cy = cyG[0];
    invfx = 1.0f / fx;
    invfy = 1.0f / fy;

    mbInitialComputations = false;
  }

  AssignFeaturesToGrid();
}

void Frame::ExtractORB(int flag, const cv::Mat &im) {
  if (flag == 0)
    (*mpORBextractor)(im, cv::Mat(), mvKeysUn, mDescriptors);
}

void Frame::ComputeBoW() {
  if (mBowVec.empty()) {
    std::vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
    mpORBvocabulary->transform(vCurrentDesc, mBowVec, mFeatVec, 4);
  }
}

void Frame::SetPose(cv::Mat Tcw) {
  mTcw = Tcw.clone();
  UpdatePoseMatrices();
}

cv::Mat Frame::GetPose() const {
  return mTcw.clone();
}

cv::Mat Frame::GetPoseInverse() const {
  return mTwc.clone();
}

cv::Mat Frame::GetRotation() const {
  return mRcw.clone();
}

cv::Mat Frame::GetTranslation() const {
  return mtcw.clone();
}

void Frame::UpdatePoseMatrices() {
  mRcw = mTcw.rowRange(0, 3).colRange(0, 3);
  mRwc = mRcw.t();
  mtcw = mTcw.rowRange(0, 3).col(3);
  mOw = -mRcw.t() * mtcw;

  mTwc = cv::Mat::eye(4, 4, mTcw.type());
  mRwc.copyTo(mTwc.rowRange(0, 3).colRange(0, 3));
  mOw.copyTo(mTwc.rowRange(0, 3).col(3));
}

/**
 * @brief Allow negative points
 * @param P
 * @param u
 * @param v
 * @return
 */
bool Frame::PointToPixel(const cv::Mat &P, float &u, float &v) {
  // 3D in camera coordinates
  const cv::Mat Pc = mRcw * P + mtcw;
  const float &PcX = Pc.at<float>(0);
  const float &PcY = Pc.at<float>(1);
  const float &PcZ = Pc.at<float>(2);

  // Check positive depth
  if (PcZ < 0.0f)
    return false;

  // Project in image and check it is not outside
  const float invz = 1.0f / PcZ;
  u = fx * PcX * invz + cx;
  v = fy * PcY * invz + cy;

  return true;
}

bool Frame::IsInFrustum(const cv::Mat &P, float &u, float &v) {

  // 3D in camera coordinates
  const cv::Mat Pc = mRcw * P + mtcw;
  const float &PcX = Pc.at<float>(0);
  const float &PcY = Pc.at<float>(1);
  const float &PcZ = Pc.at<float>(2);

  // Check positive depth
  if (PcZ < 0.0f)
    return false;

  // Project in image and check it is not outside
  const float invz = 1.0f / PcZ;
  u = fx * PcX * invz + cx;
  v = fy * PcY * invz + cy;

  if (u < mnMinX || u > mnMaxX)
    return false;
  if (v < mnMinY || v > mnMaxY)
    return false;

  return true;
}

bool Frame::IsInFrustum(MapPoint *pMP, cv::Mat &Tcw, float viewingCosLimit) {
  pMP->mbTrackInView = false;
  // 3D in absolute coordinates
  cv::Mat P = pMP->GetWorldPos();

  cv::Mat Rcw = Tcw.rowRange(0, 3).colRange(0, 3);
  cv::Mat tcw = Tcw.rowRange(0, 3).col(3);
  // 3D in camera coordinates
  const cv::Mat Pc = Rcw * P + tcw;
  const float &PcX = Pc.at<float>(0);
  const float &PcY = Pc.at<float>(1);
  const float &PcZ = Pc.at<float>(2);

  // Check positive depth
  if (PcZ < 0.0f)
    return false;

  // Project in image and check it is not outside
  const float invz = 1.0f / PcZ;
  const float u = fx * PcX * invz + cx;
  const float v = fy * PcY * invz + cy;

  if (u < mnMinX || u > mnMaxX)
    return false;
  if (v < mnMinY || v > mnMaxY)
    return false;

  const cv::Mat PO = P - mOw;
  const float dist = cv::norm(PO);
  if (!use_superpoint) {
    // Check distance is in the scale invariance region of the MapPoint
    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();

    if (dist < minDistance || dist > maxDistance)
      return false;
  }

  // Check viewing angle
  cv::Mat Pn = pMP->GetNormal();

  const float viewCos = PO.dot(Pn) / dist;

  if (viewCos < viewingCosLimit)
    return false;

  // Predict scale in the image
  const int nPredictedLevel = pMP->PredictScale(dist, this);

  // Data used by the tracking
  pMP->mbTrackInView = true;
  pMP->mTrackProjX = u;
  pMP->mTrackProjY = v;
  pMP->mnTrackScaleLevel = nPredictedLevel;
  pMP->mTrackViewCos = viewCos;

  return true;
}

bool Frame::IsInFrustum(MapPoint *pMP, float viewingCosLimit) {
  pMP->mbTrackInView = false;
  // 3D in absolute coordinates
  cv::Mat P = pMP->GetWorldPos();

  // 3D in camera coordinates
  const cv::Mat Pc = mRcw * P + mtcw;
  const float &PcX = Pc.at<float>(0);
  const float &PcY = Pc.at<float>(1);
  const float &PcZ = Pc.at<float>(2);

  // Check positive depth
  if (PcZ < 0.0f)
    return false;

  // Project in image and check it is not outside
  const float invz = 1.0f / PcZ;
  const float u = fx * PcX * invz + cx;
  const float v = fy * PcY * invz + cy;

  if (u < mnMinX || u > mnMaxX)
    return false;
  if (v < mnMinY || v > mnMaxY)
    return false;

  const cv::Mat PO = P - mOw;
  const float dist = cv::norm(PO);
  if (!use_superpoint) {
    // Check distance is in the scale invariance region of the MapPoint
    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
  
    if (dist < minDistance || dist > maxDistance)
      return false;
  }

  // Check viewing angle
  cv::Mat Pn = pMP->GetNormal();

  const float viewCos = PO.dot(Pn) / dist;

  if (viewCos < viewingCosLimit)
    return false;

  // Predict scale in the image
  const int nPredictedLevel = pMP->PredictScale(dist, this);

  // Data used by the tracking
  pMP->mbTrackInView = true;
  pMP->mTrackProjX = u;
  pMP->mTrackProjY = v;
  pMP->mnTrackScaleLevel = nPredictedLevel;
  pMP->mTrackViewCos = viewCos;

  return true;
}

bool Frame::IsInImage(const float x, const float y) const {
  return (x >= mnMinX && x < mnMaxX && y >= mnMinY && y < mnMaxY);
}

MapPoint *Frame::GetMapPoint(const size_t idx) {
  return mvpMapPoints[idx];
}

void Frame::ReplaceMapPointMatch(const size_t idx, MapPoint *pMP) {
  mvpMapPoints[idx] = pMP;
}

void Frame::EraseMapPointMatch(const size_t idx) {
  mvpMapPoints[idx] = static_cast<MapPoint *>(NULL);
}

bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY) {
  posX = round((kp.pt.x - mnMinX) * mfGridElementWidthInv);
  posY = round((kp.pt.y - mnMinY) * mfGridElementHeightInv);

  //Keypoint's coordinates are undistorted, which could cause to go out of the image
  if (posX < 0 || posX >= FRAME_GRID_COLS || posY < 0 || posY >= FRAME_GRID_ROWS)
    return false;

  return true;
}

std::set<Frame *> Frame::GetConnectedKeyFrames() {
  std::set<Frame *> s;
  for (std::map<Frame *, int>::iterator mit = mConnectedKeyFrameWeights.begin(); mit != mConnectedKeyFrameWeights.end();
       mit++)
    s.insert(mit->first);
  return s;
}

std::vector<Frame *> Frame::GetBestCovisibilityKeyFrames(const int nKFs) {
  if ((int) mvpOrderedConnectedKeyFrames.size() < nKFs)
    return mvpOrderedConnectedKeyFrames;
  else
    return std::vector<Frame *>(mvpOrderedConnectedKeyFrames.begin(), mvpOrderedConnectedKeyFrames.begin() + nKFs);
}

void Frame::AssignFeaturesToGrid() {
  int nReserve = 0.5f * N / (FRAME_GRID_COLS * FRAME_GRID_ROWS);
  for (unsigned int i = 0; i < FRAME_GRID_COLS; i++)
    for (unsigned int j = 0; j < FRAME_GRID_ROWS; j++)
      mGrid[i][j].reserve(nReserve);

  for (int i = 0; i < N; i++) {
    const cv::KeyPoint &kp = mvKeysUn[i];

    int nGridPosX, nGridPosY;
    if (PosInGrid(kp, nGridPosX, nGridPosY))
      mGrid[nGridPosX][nGridPosY].push_back(i);
  }
}

std::vector<size_t> Frame::GetFeaturesInArea(const float &x,
                                             const float &y,
                                             const float &r,
                                             const int minLevel,
                                             const int maxLevel) const {
  std::vector<size_t> vIndices;
  vIndices.reserve(N);

  const int nMinCellX = std::max(0, (int) floor((x - mnMinX - r) * mfGridElementWidthInv));
  if (nMinCellX >= FRAME_GRID_COLS)
    return vIndices;

  const int nMaxCellX = std::min((int) FRAME_GRID_COLS - 1, (int) ceil((x - mnMinX + r) * mfGridElementWidthInv));
  if (nMaxCellX < 0)
    return vIndices;

  const int nMinCellY = std::max(0, (int) floor((y - mnMinY - r) * mfGridElementHeightInv));
  if (nMinCellY >= FRAME_GRID_ROWS)
    return vIndices;

  const int nMaxCellY = std::min((int) FRAME_GRID_ROWS - 1, (int) ceil((y - mnMinY + r) * mfGridElementHeightInv));
  if (nMaxCellY < 0)
    return vIndices;

  const bool bCheckLevels = (minLevel > 0) || (maxLevel >= 0);

  for (int ix = nMinCellX; ix <= nMaxCellX; ix++) {
    for (int iy = nMinCellY; iy <= nMaxCellY; iy++) {
      const std::vector<size_t> vCell = mGrid[ix][iy];
      if (vCell.empty())
        continue;

      for (size_t j = 0, jend = vCell.size(); j < jend; j++) {
        const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
        if (bCheckLevels) {
          if (kpUn.octave < minLevel)
            continue;
          if (maxLevel >= 0)
            if (kpUn.octave > maxLevel)
              continue;
        }

        const float distx = kpUn.pt.x - x;
        const float disty = kpUn.pt.y - y;

        if (fabs(distx) < r && fabs(disty) < r)
          vIndices.push_back(vCell[j]);
      }
    }
  }

  return vIndices;
}

void Frame::UpdateConnections() {
  std::map<Frame *, int> KFcounter;
  std::map<Frame *, int> validKFcounter;
  std::vector<MapPoint *> vpMP;
  vpMP = mvpMapPoints;

  for (std::vector<MapPoint *>::iterator vit = vpMP.begin(), vend = vpMP.end(); vit != vend; vit++) {
    MapPoint *pMP = *vit;

    if (!pMP)
      continue;

    if (pMP->isBad())
      continue;

    std::map<Frame *, size_t> observations = pMP->GetObservations();

    for (std::map<Frame *, size_t>::iterator mit = observations.begin(), mend = observations.end(); mit != mend;
         mit++) {
      if (mit->first->mnId == mnId)
        continue;
      KFcounter[mit->first]++;
    }
  }

  if (KFcounter.empty()) {
    if (mnId == 1)
      return;
    else
      LOG_ASSERT(false) << " This should not happen";
  }

  //If the counter is greater than threshold add connection
  //In case no keyframe counter is over threshold add the one with maximum counter
  int nmax = 0;
  Frame *pKFmax = NULL;
  int th = 15;

  std::vector<std::pair<int, Frame *> > vPairs;
  vPairs.reserve(KFcounter.size());
  for (std::map<Frame *, int>::iterator mit = KFcounter.begin(), mend = KFcounter.end(); mit != mend; mit++) {
    if (mit->second > nmax) {
      nmax = mit->second;
      pKFmax = mit->first;
    }
    if (mit->second >= th) {
      vPairs.push_back(std::make_pair(mit->second, mit->first));
      (mit->first)->AddConnection(this, mit->second);
      validKFcounter[mit->first] = mit->second;
    }
  }

  if (vPairs.empty()) {
    vPairs.push_back(std::make_pair(nmax, pKFmax));
    pKFmax->AddConnection(this, nmax);
    validKFcounter[pKFmax] = nmax;
  }

  sort(vPairs.begin(), vPairs.end());
  std::list<Frame *> lKFs;
  std::list<int> lWs;
  for (size_t i = 0; i < vPairs.size(); i++) {
    lKFs.push_front(vPairs[i].second);
    lWs.push_front(vPairs[i].first);
  }

  {

    // mspConnectedKeyFrames = spConnectedKeyFrames;
    mConnectedKeyFrameWeights = validKFcounter;  // KFcounter
    mvpOrderedConnectedKeyFrames = std::vector<Frame *>(lKFs.begin(), lKFs.end());
    mvOrderedWeights = std::vector<int>(lWs.begin(), lWs.end());

    // LOG(INFO) << "??? " <<  mConnectedKeyFrameWeights.size() << " " << mvpOrderedConnectedKeyFrames.size() << " " << mvOrderedWeights.size();

    // we not have spanning tree

  }
}

void Frame::AddConnection(Frame *pF, const int weight) {
  if (!mConnectedKeyFrameWeights.count(pF))
    mConnectedKeyFrameWeights[pF] = weight;
  else if (mConnectedKeyFrameWeights[pF] != weight)
    mConnectedKeyFrameWeights[pF] = weight;
  else
    return;
  UpdateBestCovisibles();
}

void Frame::EraseConnection(Frame *pF) {
  bool bUpdate = false;
  {
    if (mConnectedKeyFrameWeights.count(pF)) {
      mConnectedKeyFrameWeights.erase(pF);
      bUpdate = true;
    }
  }

  if (bUpdate)
    UpdateBestCovisibles();
}

void Frame::UpdateBestCovisibles() {
  std::vector<std::pair<int, Frame *>> vPairs;
  vPairs.reserve(mConnectedKeyFrameWeights.size());
  for (std::map<Frame *, int>::iterator mit = mConnectedKeyFrameWeights.begin(), mend = mConnectedKeyFrameWeights.end();
       mit != mend; mit++)
    vPairs.push_back(std::make_pair(mit->second, mit->first));

  sort(vPairs.begin(), vPairs.end());
  std::list<Frame *> lKFs;
  std::list<int> lWs;
  for (size_t i = 0, iend = vPairs.size(); i < iend; i++) {
    lKFs.push_front(vPairs[i].second);
    lWs.push_front(vPairs[i].first);
  }

  mvpOrderedConnectedKeyFrames = std::vector<Frame *>(lKFs.begin(), lKFs.end());
  mvOrderedWeights = std::vector<int>(lWs.begin(), lWs.end());
}

/**
 * @brief
 * @param pMP ptr to MapPoint
 * @param idx idx of the mvKeysUn, mDescriptors, etc.
 */
void Frame::AddMapPoint(MapPoint *pMP, const size_t idx) {
  mvpMapPoints[idx] = pMP;
}

void Frame::SetBadFlag(DescMap *pMap) {
  if (mnId == 1) LOG_ASSERT(false);
  for (std::map<Frame *, int>::iterator mit = mConnectedKeyFrameWeights.begin(), mend = mConnectedKeyFrameWeights.end();
       mit != mend; mit++) {
    // LOG(INFO) << mit->first->mnId << " " << mConnectedKeyFrameWeights.size();
    mit->first->EraseConnection(this);
  }

  for (size_t i = 0; i < mvpMapPoints.size(); i++)
    if (mvpMapPoints[i]) mvpMapPoints[i]->EraseObservation(this);

  {
    mConnectedKeyFrameWeights.clear();
    mvpOrderedConnectedKeyFrames.clear();

    mbBad = true;
  }
  pMap->EraseKeyFrame(this);
  // mpKeyFrameDB->erase(this);
}

// convert mvpMapPoints to mFeatIdValidMapPointId
void Frame::PrepareForSaving() {
  mFeatIdValidMapPointId.clear();
  for (int idx = 0; idx < N; ++idx) {
    MapPoint *pMP = mvpMapPoints[idx];
    if (!pMP) continue;
    if (pMP->isBad()) continue;
    mFeatIdValidMapPointId[idx] = pMP->GetId();
  }
}

// convert mFeatIdValidMapPointId to mvpMapPoints
void Frame::Restore(const std::unordered_map<unsigned long, std::unique_ptr<MapPoint> > &all_map_points) {
  mvpMapPoints = std::vector<MapPoint *>(N, static_cast<MapPoint *>(NULL));
  for (auto &&it: mFeatIdValidMapPointId) {
    int feat_id = it.first;
    unsigned long map_point_id = it.second;
    mvpMapPoints[feat_id] = all_map_points.at(map_point_id).get();
  }
}

template<class Archive>
void Frame::serialize(Archive &ar, const unsigned int version) {
  ar & mnId & nNextId;
  ar & const_cast<std::string &>(mTimeStamp);
  ar & const_cast<std::vector<cv::KeyPoint> &>(mvKeysUn);

  // Image bounds and calibration
  ar & const_cast<float &>(mnMinX) & const_cast<float &>(mnMinY) & const_cast<float &>(mnMaxX)
      & const_cast<float &>(mnMaxY);
  ar & const_cast<float &>(fx) & const_cast<float &>(fy) & const_cast<float &>(cx) & const_cast<float &>(cy);
  ar & const_cast<float &>(invfx) & const_cast<float &>(invfy);

  ar & const_cast<float &>(mfGridElementWidthInv) & const_cast<float &>(mfGridElementHeightInv);
  ar & const_cast<bool &>(mbInitialComputations);
  ar & mTcw & mTwc & mRcw & mtcw & mRwc & mOw;

  // Number of KeyPoints;
  ar & const_cast<int &>(N);
  ar & mDescriptors;
  ar & mBowVec & mFeatVec;

  ar & const_cast<int &>(mnScaleLevels) & const_cast<float &>(mfScaleFactor) & const_cast<float &>(mfLogScaleFactor);
  ar & const_cast<std::vector<float> &>(mvScaleFactors) & const_cast<std::vector<float> &>(mvLevelSigma2)
      & const_cast<std::vector<float> &>(mvInvLevelSigma2);

  // FIXME: mvpMapPoints
  ar & mFeatIdValidMapPointId;
  ar & mvEnhancedMapPointGlobalIds;
  // ar & mvpMapPoints; // hope boost deal with the pointer graph well

  ar & mGrid;
  // & mConnectedKeyFrameWeights & mvpOrderedConnectedKeyFrames & mvOrderedWeights;
}
template void Frame::serialize(boost::archive::binary_iarchive &, const unsigned int);
template void Frame::serialize(boost::archive::binary_oarchive &, const unsigned int);

} // namespace dsl::relocalization