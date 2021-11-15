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
// Created by hyye on 11/11/19.
//

#ifndef DSL_CV_HELPER_H_
#define DSL_CV_HELPER_H_

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace cv_helper {

void DrawPairs(const cv::Mat &img1, const cv::Mat &img2,
               const std::vector<cv::Point2f> &pvec1,
               const std::vector<cv::Point2f> &pvec2, cv::Mat &stereo_img,
               bool draw_lines = true, int skip_count = 1,
               cv::Scalar color1 = cv::Scalar(255, 0, 0),
               cv::Scalar color2 = cv::Scalar(0, 255, 0));

void VisualizePairs(const cv::Mat &img1, const cv::Mat &img2,
                    const std::vector<cv::Point2f> &pvec1,
                    const std::vector<cv::Point2f> &pvec2,
                    bool draw_lines = true, int skip_count = 1,
                    cv::Scalar color1 = cv::Scalar(255, 0, 0),
                    cv::Scalar color2 = cv::Scalar(0, 255, 0));

}  // namespace cv_helper

#endif  // DSL_CV_HELPER_H_
