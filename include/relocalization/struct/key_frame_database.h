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

#ifndef DSL_KEY_FRAME_DATABASE_H
#define DSL_KEY_FRAME_DATABASE_H

#include "relocalization/relocalization_struct.h"
#include "relocalization/vocabulary_binary.h"

namespace dsl::relocalization {

class KeyFrameDatabase {
 public:
  KeyFrameDatabase(OrbVocabularyBinary *voc);
  void Add(Frame *pF);
  void Erase(Frame *pF);
  void Clear();
  std::vector<Frame *> DetectRelocalizationCandidates(Frame *pF);

  void BuildInvertedFile(const std::map<unsigned long, std::unique_ptr<Frame>> &all_keyframes);

  KeyFrameDatabase() {}
  void SetORBvocabulary(OrbVocabularyBinary *porbv);
 protected:
  OrbVocabularyBinary *mpVoc;

  // Inverted file
  std::vector<std::list<Frame * >> mvInvertedFile;

};

} // namespace dsl::relocalization

#endif // DSL_KEY_FRAME_DATABASE_H
