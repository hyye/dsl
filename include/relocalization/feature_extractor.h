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

#ifndef DSL_FEATURE_EXTRACTOR_H
#define DSL_FEATURE_EXTRACTOR_H

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include "util/util_common.h"
#include "DBoW2/DBoW2.h"

namespace dsl {

namespace relocalization {

void ChangeStructure(const cv::Mat &plain, std::vector<cv::Mat> &out);

class FeatureExtractor {
 public:
  virtual void operator()(cv::InputArray image, cv::InputArray mask,
                          std::vector<cv::KeyPoint> &keypoints,
                          cv::OutputArray descriptors) = 0;
};

class NaiveFeatureExtractor : public FeatureExtractor {
 public:
  NaiveFeatureExtractor() {
    orb = cv::ORB::create();
  }
  void ExtractFeatures(const cv::Mat &image,
                       std::vector<cv::KeyPoint> &keypoints,
                       std::vector<cv::Mat> &features,
                       const cv::Mat &mask = cv::Mat());
  void operator()(cv::InputArray image, cv::InputArray mask,
                  std::vector<cv::KeyPoint> &keypoints,
                  cv::OutputArray descriptors);

  cv::Ptr<cv::ORB> orb;
};

class ExtractorNode {
 public:
  ExtractorNode() : bNoMore(false) {}

  void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);

  std::vector<cv::KeyPoint> vKeys;
  cv::Point2i UL, UR, BL, BR;
  std::list<ExtractorNode>::iterator lit;
  bool bNoMore;
};

class ORBextractor : public FeatureExtractor {
 public:

  enum { HARRIS_SCORE = 0, FAST_SCORE = 1 };

  ORBextractor(int _nfeatures, float _scaleFactor, int _nlevels,
               int _iniThFAST, int _minThFAST);

  ~ORBextractor() {}

  // Compute the ORB features and descriptors on an image.
  // ORB are dispersed on the image using an octree.
  // Mask is ignored in the current implementation.
  virtual void operator()(cv::InputArray image, cv::InputArray mask,
                          std::vector<cv::KeyPoint> &keypoints,
                          cv::OutputArray descriptors);

  int inline GetLevels() {
    return nlevels;
  }

  float inline GetScaleFactor() {
    return scaleFactor;
  }

  std::vector<float> inline GetScaleFactors() {
    return mvScaleFactor;
  }

  std::vector<float> inline GetInverseScaleFactors() {
    return mvInvScaleFactor;
  }

  std::vector<float> inline GetScaleSigmaSquares() {
    return mvLevelSigma2;
  }

  std::vector<float> inline GetInverseScaleSigmaSquares() {
    return mvInvLevelSigma2;
  }

  std::vector<cv::Mat> mvImagePyramid;

 protected:

  void ComputePyramid(cv::Mat image);
  void ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint> > &allKeypoints);
  std::vector<cv::KeyPoint> DistributeOctTree(const std::vector<cv::KeyPoint> &vToDistributeKeys,
                                              const int &minX, const int &maxX,
                                              const int &minY, const int &maxY,
                                              const int &nFeatures,
                                              const int &level);

  void ComputeKeyPointsOld(std::vector<std::vector<cv::KeyPoint> > &allKeypoints);
  std::vector<cv::Point> pattern;

  int nfeatures;
  double scaleFactor;
  int nlevels;
  int iniThFAST;
  int minThFAST;

  std::vector<int> mnFeaturesPerLevel;

  std::vector<int> umax;

  std::vector<float> mvScaleFactor;
  std::vector<float> mvInvScaleFactor;
  std::vector<float> mvLevelSigma2;
  std::vector<float> mvInvLevelSigma2;
};

} // namespace relocalization

} // namespace dsl

#endif // DSL_FEATURE_EXTRACTOR_H
