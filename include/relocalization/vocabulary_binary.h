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

#ifndef DSL_VOCABULARY_BINARY_H
#define DSL_VOCABULARY_BINARY_H

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "relocalization/boost_archiver.h"
#include "DBoW2/DBoW2.h"

namespace dsl {

namespace relocalization {

template<class TDescriptor, class F>
class VocabularyBinaryBoost : public DBoW2::TemplatedVocabulary<TDescriptor, F> {
 public:
  bool loadFromTextFile(const std::string &filename);
  bool loadFromBinaryFile(const std::string &filename);
  void saveToBinaryFile(const std::string &filename);

  struct NodeSerializable : public DBoW2::TemplatedVocabulary<TDescriptor, F>::Node {
    NodeSerializable() {}
    NodeSerializable(typename DBoW2::TemplatedVocabulary<TDescriptor, F>::Node &n) {
      this->id = n.id;
      this->weight = n.weight;
      this->children = n.children;
      this->parent = n.parent;
      this->descriptor = n.descriptor; // .clone();
      this->word_id = n.word_id;
    }

   private:
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
      ar & this->id;
      ar & this->weight;
      ar & this->children;
      ar & this->parent;
      ar & this->descriptor;
      ar & this->word_id;
    }
  };

  int GetK() { return this->m_k; }
  int GetL() { return this->m_L; }
  DBoW2::WeightingType GetWeighting() { return this->m_weighting; }
  DBoW2::ScoringType GetScoring() { return this->m_scoring; }
  std::vector<typename DBoW2::TemplatedVocabulary<TDescriptor, F>::Node> &GetNodes() { return this->m_nodes; }

 protected:
  std::vector<NodeSerializable> m_nodes_serializable;

 private:

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive &ar, const unsigned int version) {
    ar & this->m_k;
    ar & this->m_L;
    ar & this->m_scoring;
    ar & this->m_weighting;
    ar & this->m_nodes_serializable;
  }
};

template
class VocabularyBinaryBoost<DBoW2::FORB::TDescriptor, DBoW2::FORB>;

template
class VocabularyBinaryBoost<DBoW2::FBrief::TDescriptor, DBoW2::FBrief>;

// Adapted from https://github.com/Jiankai-Sun/ORB_SLAM2_Enhanced
template<class TDescriptor, class F>
class VocabularyBinary : public DBoW2::TemplatedVocabulary<TDescriptor, F> {
 public:
  bool loadFromBinaryFile(const std::string &filename);
  bool loadFromTextFile(const std::string &filename);
  void saveToBinaryFile(const std::string &filename) const;

  void transformToNodeId(const TDescriptor &feature,
                         DBoW2::NodeId &nid, int levelsup) const;

  int GetK() { return this->m_k; }
  int GetL() { return this->m_L; }
  DBoW2::WeightingType GetWeighting() { return this->m_weighting; }
  DBoW2::ScoringType GetScoring() { return this->m_scoring; }
  std::vector<typename DBoW2::TemplatedVocabulary<TDescriptor, F>::Node> &GetNodes() { return this->m_nodes; }
};

template<class TDescriptor, class F>
bool VocabularyBinary<TDescriptor, F>::loadFromTextFile(const std::string &filename) {
  std::ifstream f;
  f.open(filename.c_str());

  if (f.eof())
    return false;

  this->m_words.clear();
  this->m_nodes.clear();

  std::string s;
  getline(f, s);
  std::stringstream ss;
  ss << s;
  ss >> this->m_k;
  ss >> this->m_L;
  int n1, n2;
  ss >> n1;
  ss >> n2;

  if (this->m_k < 0 || this->m_k > 20 || this->m_L < 1 || this->m_L > 10 || n1 < 0 || n1 > 5 || n2 < 0 || n2 > 3) {
    std::cerr << "Vocabulary loading failure: This is not a correct text file!" << std::endl;
    return false;
  }

  this->m_scoring = (DBoW2::ScoringType) n1;
  this->m_weighting = (DBoW2::WeightingType) n2;
  this->createScoringObject();

  // nodes
  int expected_nodes =
      (int) ((pow((double) this->m_k, (double) this->m_L + 1) - 1) / (this->m_k - 1));
  this->m_nodes.reserve(expected_nodes);

  this->m_words.reserve(pow((double) this->m_k, (double) this->m_L + 1));

  this->m_nodes.resize(1);
  this->m_nodes[0].id = 0;
  while (!f.eof()) {
    std::string snode;
    getline(f, snode);
    std::stringstream ssnode;
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

    std::stringstream ssd;
    for (int iD = 0; iD < F::L; iD++) {
      std::string sElement;
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

template<class TDescriptor, class F>
bool VocabularyBinary<TDescriptor, F>::loadFromBinaryFile(const std::string &filename) {
  std::fstream f;
  f.open(filename.c_str(), std::ios_base::in | std::ios::binary);
  unsigned int nb_nodes, size_node;
  f.read((char *) &nb_nodes, sizeof(nb_nodes));
  f.read((char *) &size_node, sizeof(size_node));
  f.read((char *) &this->m_k, sizeof(this->m_k));
  f.read((char *) &this->m_L, sizeof(this->m_L));
  f.read((char *) &this->m_scoring, sizeof(this->m_scoring));
  f.read((char *) &this->m_weighting, sizeof(this->m_weighting));
  this->createScoringObject();

  this->m_words.clear();
  this->m_words.reserve(pow((double) this->m_k, (double) this->m_L + 1));
  this->m_nodes.clear();
  this->m_nodes.resize(nb_nodes); // nb_nodes + 1
  this->m_nodes[0].id = 0;
  char *buf = new char[size_node];
  int nid = 1;
  f.read(buf, size_node);
  while (!f.eof()) {
    this->m_nodes[nid].id = nid;
    // FIXME
    const int *ptr = (int *) buf;
    this->m_nodes[nid].parent = *ptr;
    //m_nodes[nid].parent = *(const int*)buf;
    this->m_nodes[this->m_nodes[nid].parent].children.push_back(nid);
    this->m_nodes[nid].descriptor = cv::Mat(1, F::L, CV_8U); //F::L
    memcpy(this->m_nodes[nid].descriptor.data, buf + 4, F::L); //F::L
    this->m_nodes[nid].weight = *(float *) (buf + 4 + F::L); // F::L
    if (buf[8 + F::L]) { // is leaf //F::L
      int wid = this->m_words.size();
      this->m_words.resize(wid + 1);
      this->m_nodes[nid].word_id = wid;
      this->m_words[wid] = &this->m_nodes[nid];
    } else
      this->m_nodes[nid].children.reserve(this->m_k);
    nid += 1;
    f.read(buf, size_node); // read in the end of the loop!
  }
  f.close();

  delete[] buf;
  return true;
}

template<class TDescriptor, class F>
void VocabularyBinary<TDescriptor, F>::saveToBinaryFile(const std::string &filename) const {
  std::fstream f;
  f.open(filename.c_str(), std::ios_base::out | std::ios::binary);
  unsigned int nb_nodes = this->m_nodes.size();
  float _weight;
  unsigned int
      size_node = sizeof(this->m_nodes[0].parent) + F::L * sizeof(char) + sizeof(_weight) + sizeof(bool); //F::L
  f.write((char *) &nb_nodes, sizeof(nb_nodes));
  f.write((char *) &size_node, sizeof(size_node));
  f.write((char *) &this->m_k, sizeof(this->m_k));
  f.write((char *) &this->m_L, sizeof(this->m_L));
  f.write((char *) &this->m_scoring, sizeof(this->m_scoring));
  f.write((char *) &this->m_weighting, sizeof(this->m_weighting));
  for (size_t i = 1; i < nb_nodes; i++) {
    const typename VocabularyBinary<TDescriptor, F>::Node &node = this->m_nodes[i];
    f.write((char *) &node.parent, sizeof(node.parent));
    f.write((char *) node.descriptor.data, F::L);//F::L
    _weight = node.weight;
    f.write((char *) &_weight, sizeof(_weight));
    bool is_leaf = node.isLeaf();
    f.write((char *) &is_leaf, sizeof(is_leaf)); // i put this one at the end for alignement....
  }
  f.close();
}

template<class TDescriptor, class F>
void VocabularyBinary<TDescriptor, F>::transformToNodeId(const TDescriptor &feature,
                                                         DBoW2::NodeId &nid,
                                                         int levelsup) const {
  DBoW2::WordId word_id;
  this->DBoW2::TemplatedVocabulary<TDescriptor, F>::transform(feature, word_id);

  nid = this->DBoW2::TemplatedVocabulary<TDescriptor, F>::getParentNode(word_id, levelsup);
}

/// ORB Vocabulary
typedef VocabularyBinary<DBoW2::FORB::TDescriptor, DBoW2::FORB> OrbVocabularyBinary;

} // namespace relocalization

} // namespace dsl

#endif // DSL_VOCABULARY_BINARY_H
