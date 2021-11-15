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
// Created by hyye on 7/13/20.
//

#include "relocalization/struct/key_frame_database.h"

using namespace std;

namespace dsl::relocalization {

KeyFrameDatabase::KeyFrameDatabase(OrbVocabularyBinary *voc) : mpVoc(voc) {
  mvInvertedFile.resize(voc->size());
}

void KeyFrameDatabase::Add(Frame *pF) {
  for (DBoW2::BowVector::const_iterator vit = pF->mBowVec.begin(), vend = pF->mBowVec.end(); vit != vend; vit++)
    mvInvertedFile[vit->first].push_back(pF);
}

void KeyFrameDatabase::Erase(Frame *pF) {
  // Erase elements in the Inverse File for the entry
  for (DBoW2::BowVector::const_iterator vit = pF->mBowVec.begin(), vend = pF->mBowVec.end(); vit != vend; vit++) {
    // List of keyframes that share the word
    std::list<Frame *> &lKFs = mvInvertedFile[vit->first];

    for (std::list<Frame *>::iterator lit = lKFs.begin(), lend = lKFs.end(); lit != lend; lit++) {
      if (pF == *lit) {
        lKFs.erase(lit);
        break;
      }
    }
  }
}

void KeyFrameDatabase::Clear() {
  mvInvertedFile.clear();
  mvInvertedFile.resize(mpVoc->size());
}

std::vector<Frame *> KeyFrameDatabase::DetectRelocalizationCandidates(Frame *pF) {
  list<Frame *> lKFsSharingWords;

  // Search all keyframes that share a word with current frame
  {

    for (DBoW2::BowVector::const_iterator vit = pF->mBowVec.begin(), vend = pF->mBowVec.end(); vit != vend; vit++) {
      list<Frame *> &lKFs = mvInvertedFile[vit->first];

      for (list<Frame *>::iterator lit = lKFs.begin(), lend = lKFs.end(); lit != lend; lit++) {
        Frame *pKFi = *lit;
        if (pKFi->mnRelocQuery != pF->mnId) {
          pKFi->mnRelocWords = 0;
          pKFi->mnRelocQuery = pF->mnId;
          lKFsSharingWords.push_back(pKFi);
        }
        pKFi->mnRelocWords++;
      }
    }
  }

  if (lKFsSharingWords.empty())
    return vector<Frame *>();

  // Only compare against those keyframes that share enough words
  int maxCommonWords = 0;
  for (list<Frame *>::iterator lit = lKFsSharingWords.begin(), lend = lKFsSharingWords.end(); lit != lend; lit++) {
    if ((*lit)->mnRelocWords > maxCommonWords)
      maxCommonWords = (*lit)->mnRelocWords;
  }

  int minCommonWords = maxCommonWords * 0.8f;

  list<pair<float, Frame *> > lScoreAndMatch;

  int nscores = 0;

  // Compute similarity score.
  for (list<Frame *>::iterator lit = lKFsSharingWords.begin(), lend = lKFsSharingWords.end(); lit != lend; lit++) {
    Frame *pKFi = *lit;

    if (pKFi->mnRelocWords > minCommonWords) {
      nscores++;
      float si = mpVoc->score(pF->mBowVec, pKFi->mBowVec);
      pKFi->mRelocScore = si;
      lScoreAndMatch.push_back(make_pair(si, pKFi));
      LOG(INFO) << "mnRelocWords: " << pKFi->mnRelocWords;
    }
  }

  if (lScoreAndMatch.empty())
    return vector<Frame *>();

  list<pair<float, Frame *> > lAccScoreAndMatch;
  float bestAccScore = 0;

  // Lets now accumulate score by covisibility
  for (list<pair<float, Frame *> >::iterator it = lScoreAndMatch.begin(), itend = lScoreAndMatch.end(); it != itend;
       it++) {
    Frame *pKFi = it->second;
    vector<Frame *> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

    float bestScore = it->first;
    float accScore = bestScore;
    Frame *pBestKF = pKFi;
    for (vector<Frame *>::iterator vit = vpNeighs.begin(), vend = vpNeighs.end(); vit != vend; vit++) {
      Frame *pKF2 = *vit;
      if (pKF2->mnRelocQuery != pF->mnId)
        continue;

      accScore += pKF2->mRelocScore;
      if (pKF2->mRelocScore > bestScore) {
        pBestKF = pKF2;
        bestScore = pKF2->mRelocScore;
      }

    }
    lAccScoreAndMatch.push_back(make_pair(accScore, pBestKF));
    if (accScore > bestAccScore)
      bestAccScore = accScore;
  }

  // Return all those keyframes with a score higher than 0.75*bestScore
  float minScoreToRetain = 0.75f * bestAccScore;
  set<Frame *> spAlreadyAddedKF;
  vector<Frame *> vpRelocCandidates;
  vpRelocCandidates.reserve(lAccScoreAndMatch.size());
  for (list<pair<float, Frame *> >::iterator it = lAccScoreAndMatch.begin(), itend = lAccScoreAndMatch.end();
       it != itend; it++) {
    const float &si = it->first;
    if (si > minScoreToRetain) {
      Frame *pKFi = it->second;
      if (!spAlreadyAddedKF.count(pKFi)) {
        vpRelocCandidates.push_back(pKFi);
        spAlreadyAddedKF.insert(pKFi);
      }
      if (spAlreadyAddedKF.size() >= 3) break;
    }
  }

  return vpRelocCandidates;
}

void KeyFrameDatabase::BuildInvertedFile(const std::map<unsigned long,
                                                        std::unique_ptr<Frame>> &all_keyframes) {
  for (auto &&it : all_keyframes) {
    Frame *pMP = it.second.get();
    for (DBoW2::BowVector::const_iterator vit = pMP->mBowVec.begin(),
             vend = pMP->mBowVec.end();
         vit != vend; vit++)
      mvInvertedFile[vit->first].push_back(pMP);
  }
}

void KeyFrameDatabase::SetORBvocabulary(OrbVocabularyBinary *porbv) {
  mpVoc = porbv;
  mvInvertedFile.clear();
  mvInvertedFile.resize(mpVoc->size());
}

} // namespace dsl::relocalization