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
// Created by hyye on 7/10/20.
//

#ifndef DSL_DRAWER_H
#define DSL_DRAWER_H

#include "relocalization/relocalization_struct.h"

namespace dsl::relocalization {

void AddTextToImage(const std::string &s, cv::Mat &im, const int r, const int g, const int b);

void ConvertMatchesToKpts(Frame *const pF,
                          const std::vector<MapPoint *> &vpMapPointMatches,
                          cv::Mat &Tcw,
                          std::vector<cv::KeyPoint> &kpts);

void ConvertMatches(Frame *const pF,
                    Frame *const pKF,
                    const std::vector<MapPoint *> &vpMapPointMatches,
                    std::vector<cv::KeyPoint> &kpts1,
                    std::vector<cv::KeyPoint> &kpts2,
                    std::vector<cv::DMatch> &matches1to2);

void DrawMatchesPair(const std::vector<std::pair<cv::KeyPoint, cv::KeyPoint>> &kpt_pairs,
                     const cv::Mat &img1,
                     const cv::Mat &img2,
                     cv::Mat &outImg);

void DrawMapPointMatches(const Frame *const pF,
                         const std::vector<MapPoint *> &vpMapPointMatches,
                         const cv::Mat &inImg,
                         cv::Mat &outImg);
void DrawImagePair(const cv::Mat &img1, const cv::Mat &img2, cv::Mat &outImg);
cv::Mat MakeCanvas(std::vector<cv::Mat> &vecMat, int windowHeight, int nRows);

} // namespace dsl::relocalization


#endif // DSL_DRAWER_H
