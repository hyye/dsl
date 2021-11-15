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

#include "tool/cv_helper.h"

namespace cv_helper {

void DrawPairs(const cv::Mat &img1, const cv::Mat &img2,
               const std::vector<cv::Point2f> &pvec1,
               const std::vector<cv::Point2f> &pvec2, cv::Mat &stereo_img,
               bool draw_lines, int skip_count, cv::Scalar color1,
               cv::Scalar color2) {
  assert(img1.size() == img2.size());
  assert(pvec1.size() == pvec2.size());

  for (int i = 0; i < pvec1.size(); i += skip_count) {
    cv::Point2f left = cv::Point2f(pvec1[i].x, pvec1[i].y);
    cv::Point2f right = (cv::Point2f(pvec2[i].x, pvec2[i].y) +
                         cv::Point2f((float)img1.cols, 0.f));
    if (draw_lines) {
      line(stereo_img, left, right, cv::Scalar(255, 0, 0));
    }

    circle(stereo_img, left, 1, color1, -1, cv::FILLED);
    circle(stereo_img, right, 1, color2, -1, cv::FILLED);
  }
}

void VisualizePairs(const cv::Mat &img1, const cv::Mat &img2,
                    const std::vector<cv::Point2f> &pvec1,
                    const std::vector<cv::Point2f> &pvec2, bool draw_lines,
                    int skip_count, cv::Scalar color1, cv::Scalar color2) {
  cv::Mat stereo_img(
      cv::Mat(img1.rows, img1.cols + img2.cols, CV_8UC3, cv::Scalar(0, 0, 0)));
  img1.copyTo(stereo_img(cv::Rect(0, 0, img1.cols, img1.rows)));
  img2.copyTo(stereo_img(cv::Rect(img2.cols, 0, img2.cols, img2.rows)));
  DrawPairs(img1, img2, pvec1, pvec2, stereo_img, draw_lines, skip_count,
            color1, color2);

  cv::namedWindow("vis", cv::WINDOW_NORMAL);
  cv::imshow("vis", stereo_img);
  cv::resizeWindow("vis", 640 * 2, 480);
}

}