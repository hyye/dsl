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
// Created by hyye on 6/30/20.
//

#include "relocalization/vocabulary_binary.h"

using namespace std;
using namespace DBoW2;

namespace dsl {

namespace relocalization {

template<class TDescriptor, class F>
bool VocabularyBinaryBoost<TDescriptor, F>::loadFromTextFile(const std::string &filename) {
  ifstream f;
  f.open(filename.c_str());

  if (f.eof())
    return false;

  this->m_words.clear();
  this->m_nodes.clear();

  string s;
  getline(f, s);
  stringstream ss;
  ss << s;
  ss >> this->m_k;
  ss >> this->m_L;
  int n1, n2;
  ss >> n1;
  ss >> n2;

  if (this->m_k < 0 || this->m_k > 20 || this->m_L < 1 || this->m_L > 10 || n1 < 0 || n1 > 5 || n2 < 0 || n2 > 3) {
    std::cerr << "Vocabulary loading failure: This is not a correct text file!" << endl;
    return false;
  }

  this->m_scoring = (ScoringType) n1;
  this->m_weighting = (WeightingType) n2;
  this->createScoringObject();

  // nodes
  int expected_nodes =
      (int) ((pow((double) this->m_k, (double) this->m_L + 1) - 1) / (this->m_k - 1));
  this->m_nodes.reserve(expected_nodes);

  this->m_words.reserve(pow((double) this->m_k, (double) this->m_L + 1));

  this->m_nodes.resize(1);
  this->m_nodes[0].id = 0;
  while (!f.eof()) {
    string snode;
    getline(f, snode);
    stringstream ssnode;
    ssnode << snode;

    int nid = this->m_nodes.size();
    this->m_nodes.resize(this->m_nodes.size() + 1);
    this->m_nodes[nid].id = nid;

    int pid;
    ssnode >> pid;
    this->m_nodes[nid].parent = pid;
    this->m_nodes[pid].children.push_back(nid);

    int nIsLeaf;
    ssnode >> nIsLeaf;

    stringstream ssd;
    for (int iD = 0; iD < F::L; iD++) {
      string sElement;
      ssnode >> sElement;
      ssd << sElement << " ";
    }
    F::fromString(this->m_nodes[nid].descriptor, ssd.str());

    ssnode >> this->m_nodes[nid].weight;

    if (nIsLeaf > 0) {
      int wid = this->m_words.size();
      this->m_words.resize(wid + 1);

      this->m_nodes[nid].word_id = wid;
      this->m_words[wid] = &this->m_nodes[nid];
    } else {
      this->m_nodes[nid].children.reserve(this->m_k);
    }
  }

  return true;

}

// template<class TDescriptor, class F> template <class Archive>
// void VocabularyBinaryBoost<TDescriptor, F>::serialize(Archive &ar, const unsigned int version) {
//   ar & this->m_k;
//   ar & this->m_L;
//   ar & this->m_scoring;
//   ar & this->m_weighting;
//   ar & this->m_nodes;
// }

template<class TDescriptor, class F>
void VocabularyBinaryBoost<TDescriptor, F>::saveToBinaryFile(const std::string &filename) {
  // create and open a character archive for output
  std::ofstream ofs(filename);

  // save data to archive
  {
    for (auto&& n : this->m_nodes) {
      NodeSerializable n_serializable = n;
      this->m_nodes_serializable.push_back(n_serializable);
    }
    boost::archive::binary_oarchive oa(ofs);
    // write class instance to archive
    oa << *this;
    // archive and stream closed when destructors are called
  }
}

template<class TDescriptor, class F>
bool VocabularyBinaryBoost<TDescriptor, F>::loadFromBinaryFile(const std::string &filename) {
  std::ifstream ifs(filename);
  boost::archive::binary_iarchive ia(ifs);
  VocabularyBinaryBoost<TDescriptor, F> voc;
  ia >> voc;

  voc.m_nodes.clear();
  for (auto&& n_serializable : voc.m_nodes_serializable) {
    typename VocabularyBinaryBoost<TDescriptor, F>::Node n = n_serializable;
    // NOTE: not necessary, since Ptr???
    n.descriptor = n_serializable.descriptor; // .clone();
    voc.m_nodes.push_back(n);
  }

  this->m_k = voc.m_k;
  this->m_L = voc.m_L;
  this->m_scoring = voc.m_scoring;
  this->m_weighting = voc.m_weighting;

  this->createScoringObject();

  this->m_nodes.clear();
  this->m_words.clear();

  this->m_nodes = voc.m_nodes;
  this->createWords();
  return true;
}

} // namespace relocalization

} // namespace dsl