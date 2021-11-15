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
// Created by hyye on 7/5/20.
//

#ifndef DSL_BOOST_ARCHIVER_H
#define DSL_BOOST_ARCHIVER_H

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/bitset.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/set.hpp>
#include <boost/serialization/list.hpp>
#include <boost/serialization/split_free.hpp>

#include <opencv2/core.hpp>

#include "DBoW2/DBoW2.h"

BOOST_SERIALIZATION_SPLIT_FREE(::cv::Mat)
namespace boost {
namespace serialization {

/* serialization for DBoW2 BowVector */
template<class Archive>
void serialize(Archive &ar, DBoW2::BowVector &BowVec, const unsigned int file_version) {
  ar & boost::serialization::base_object<std::map<DBoW2::WordId, DBoW2::WordValue>>(BowVec);
}
/* serialization for DBoW2 FeatureVector */
template<class Archive>
void serialize(Archive &ar, DBoW2::FeatureVector &FeatVec, const unsigned int file_version) {
  ar & boost::serialization::base_object<std::map<DBoW2::NodeId, std::vector<unsigned int> >>(FeatVec);
}

/* serialization for CV KeyPoint */
template<class Archive>
void serialize(Archive &ar, ::cv::KeyPoint &kf, const unsigned int file_version) {
  ar & kf.angle;
  ar & kf.class_id;
  ar & kf.octave;
  ar & kf.response;
  ar & kf.size;
  ar & kf.pt.x;
  ar & kf.pt.y;
}

/*** Mat ***/
template<class Archive>
void save(Archive &ar, const cv::Mat &m, const unsigned int version) {
  cv::Mat m_ = m;
  if (!m.isContinuous())
    m_ = m.clone();

  size_t elemSize = m_.elemSize(), elemType = m_.type();

  ar & m_.cols;
  ar & m_.rows;
  ar & elemSize;
  ar & elemType; // element type.
  size_t dataSize = m_.cols * m_.rows * m_.elemSize();

  //cout << "Writing matrix data rows, cols, elemSize, type, datasize: (" << m.rows << "," << m.cols << "," << m.elemSize() << "," << m.type() << "," << dataSize << ")" << endl;

  ar & boost::serialization::make_array(m_.ptr(), dataSize);
}

template<class Archive>
void load(Archive &ar, cv::Mat &m, const unsigned int version) {
  int cols, rows;
  size_t elemSize, elemType;

  ar & cols;
  ar & rows;
  ar & elemSize;
  ar & elemType;

  m.create(rows, cols, elemType);
  size_t dataSize = m.cols * m.rows * elemSize;

  //cout << "reading matrix data rows, cols, elemSize, type, datasize: (" << m.rows << "," << m.cols << "," << m.elemSize() << "," << m.type() << "," << dataSize << ")" << endl;

  ar & boost::serialization::make_array(m.ptr(), dataSize);
}

} // serialization

} // namespace boost

#endif // DSL_BOOST_ARCHIVER_H
