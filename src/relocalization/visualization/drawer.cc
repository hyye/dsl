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

#include "relocalization/visualization/drawer.h"

namespace dsl::relocalization {

void AddTextToImage(const std::string &s, cv::Mat &im, const int r, const int g, const int b) {
  int l = 10;
  //imText.rowRange(im.rows-imText.rows,imText.rows) = cv::Mat::zeros(textSize.height+10,im.cols,im.type());
  cv::putText(im, s, cv::Point(l, im.rows - l), cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(255, 255, 255), 2, 8);
  cv::putText(im, s, cv::Point(l - 1, im.rows - l), cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(255, 255, 255), 2, 8);
  cv::putText(im, s, cv::Point(l + 1, im.rows - l), cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(255, 255, 255), 2, 8);
  cv::putText(im, s, cv::Point(l - 1, im.rows - (l - 1)), cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(255, 255, 255), 2, 8);
  cv::putText(im, s, cv::Point(l, im.rows - (l - 1)), cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(255, 255, 255), 2, 8);
  cv::putText(im, s, cv::Point(l + 1, im.rows - (l - 1)), cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(255, 255, 255), 2, 8);
  cv::putText(im, s, cv::Point(l - 1, im.rows - (l + 1)), cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(255, 255, 255), 2, 8);
  cv::putText(im, s, cv::Point(l, im.rows - (l + 1)), cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(255, 255, 255), 2, 8);
  cv::putText(im, s, cv::Point(l + 1, im.rows - (l + 1)), cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(255, 255, 255), 2, 8);

  cv::putText(im, s, cv::Point(l, im.rows - l), cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(r, g, b), 2, 8);
}

void ConvertMatchesToKpts(Frame *const pF,
                          const std::vector<MapPoint *> &vpMapPointMatches,
                          cv::Mat &Tcw,
                          std::vector<cv::KeyPoint> &kpts) {
  kpts.clear();

  cv::Mat Rcw = Tcw.rowRange(0, 3).colRange(0, 3);
  cv::Mat tcw = Tcw.rowRange(0, 3).col(3);

  for (int feat_id = 0; feat_id < vpMapPointMatches.size(); ++feat_id) {
    MapPoint *pMP = vpMapPointMatches[feat_id];
    if (pMP && !pMP->isBad()) {
      cv::Mat worldPos = pMP->GetWorldPos();
      float u, v;

      const cv::Mat Pc = Rcw * worldPos + tcw;
      const float &PcX = Pc.at<float>(0);
      const float &PcY = Pc.at<float>(1);
      const float &PcZ = Pc.at<float>(2);

      // Check positive depth
      if (PcZ < 0.0f) continue;

      // Project in image and check it is not outside
      const float invz = 1.0f / PcZ;
      u = pF->fx * PcX * invz + pF->cx;
      v = pF->fy * PcY * invz + pF->cy;

      kpts.emplace_back(u, v, 31, 0);
    }
  }
}

void DrawMatchesPair(const std::vector<std::pair<cv::KeyPoint, cv::KeyPoint>> &kpt_pairs,
                     const cv::Mat &img1,
                     const cv::Mat &img2,
                     cv::Mat &outImg) {
  std::vector<cv::KeyPoint> kps1, kps2;
  std::vector<cv::DMatch> matches1to2;
  for (auto &&pair : kpt_pairs) {
    cv::KeyPoint kp1 = pair.first;
    cv::KeyPoint kp2 = pair.second;
    cv::DMatch match;
    match.trainIdx = kps1.size();
    match.queryIdx = kps2.size();
    kps1.push_back(kp1);
    kps2.push_back(kp2);
    matches1to2.push_back(match);
  }

  cv::drawMatches(img1, kps1, img2, kps2, matches1to2, outImg);
}

void ConvertMatches(Frame *const pF,
                    Frame *const pKF,
                    const std::vector<MapPoint *> &vpMapPointMatches,
                    std::vector<cv::KeyPoint> &kpts1,
                    std::vector<cv::KeyPoint> &kpts2,
                    std::vector<cv::DMatch> &matches1to2) {
  std::vector<cv::KeyPoint> not_found_kpts1;
  kpts1.clear();
  kpts2.clear();
  matches1to2.clear();
  for (int feat_id = 0; feat_id < vpMapPointMatches.size(); ++feat_id) {
    MapPoint *pMP = vpMapPointMatches[feat_id];
    if (pMP && !pMP->isBad()) {
      cv::DMatch match;
      cv::KeyPoint kpt = pF->mvKeysUn[feat_id];;
      cv::KeyPoint kf_kpt;
      if (pMP->GetObservations().count(pKF)) {
        size_t kf_feat_id = pMP->GetObservations().at(pKF);
        kf_kpt = pKF->mvKeysUn[kf_feat_id];
      } else {
        cv::Mat worldPos = pMP->GetWorldPos();
        float u, v;
        if (pKF->IsInFrustum(worldPos, u, v)) {
          kf_kpt = cv::KeyPoint(u, v, 31, 0); // set to default size and angle for visualization
        } else {
          not_found_kpts1.push_back(kpt);
          continue;
        }
      }
      match.trainIdx = kpts1.size();
      match.queryIdx = kpts2.size();
      kpts1.push_back(kpt);
      kpts2.push_back(kf_kpt);
      matches1to2.push_back(match);
    }
  }
  kpts1.insert(kpts1.end(), not_found_kpts1.begin(), not_found_kpts1.end());
  // LOG(INFO) << "??? not_found_kpts1 " << not_found_kpts1.size();
}

void DrawImagePair(const cv::Mat &img1, const cv::Mat &img2, cv::Mat &outImg) {
  outImg = cv::Mat(std::max(img1.size().height, img2.size().height), img1.size().width + img2.size().width, CV_8UC3);

  cv::Mat img1_rgb = img1, img2_rgb = img2;
  if (img1.channels() == 1) cv::cvtColor(img1, img1_rgb, CV_GRAY2BGR);
  if (img2.channels() == 1) cv::cvtColor(img2, img2_rgb, CV_GRAY2BGR);

  cv::Mat left_roi(outImg, cv::Rect(0, 0, img1.size().width, img1.size().height));
  img1_rgb.copyTo(left_roi);
  cv::Mat right_roi(outImg, cv::Rect(img1.size().width, 0, img2.size().width, img2.size().height));
  img2_rgb.copyTo(right_roi);
}

void DrawMapPointMatches(const Frame *const pF,
                         const std::vector<MapPoint *> &vpMapPointMatches,
                         const cv::Mat &inImg,
                         cv::Mat &outImg) {
  std::vector<cv::KeyPoint> kps1, kps2;
  std::vector<cv::DMatch> matches1to2;
  for (int i = 0; i < pF->mvKeysUn.size(); ++i) {
    const cv::KeyPoint &kp = pF->mvKeysUn[i];
    MapPoint *pMP = vpMapPointMatches[i];
    if (pMP) {
      cv::Mat Rcw = pF->GetRotation();
      cv::Mat tcw = pF->GetTranslation();

      const float &fx = pF->fx;
      const float &fy = pF->fy;
      const float &cx = pF->cx;
      const float &cy = pF->cy;

      cv::Mat p3Dw = pMP->GetWorldPos();
      cv::Mat p3Dc = Rcw * p3Dw + tcw;
      if (p3Dc.at<float>(2) < 0.0f)
        continue;
      const float invz = 1 / p3Dc.at<float>(2);
      const float x = p3Dc.at<float>(0) * invz;
      const float y = p3Dc.at<float>(1) * invz;

      const float u = fx * x + cx;
      const float v = fy * y + cy;

      // Point must be inside the image
      if (!pF->IsInImage(u, v))
        continue;

      cv::KeyPoint kp_from_3d = kp;
      kp_from_3d.pt.x = u;
      kp_from_3d.pt.y = v;
      cv::DMatch match;
      match.trainIdx = kps1.size();
      match.queryIdx = kps2.size();
      kps1.push_back(kp);
      kps2.push_back(kp_from_3d);
      matches1to2.push_back(match);
    }
  }
  cv::drawMatches(inImg, kps1, inImg, kps2, matches1to2, outImg);
}

/**
    * @brief makeCanvas Makes composite image from the given images
    * @param vecMat Vector of Images.
    * @param windowHeight The height of the new composite image to be formed.
    * @param nRows Number of rows of images. (Number of columns will be calculated
    *              depending on the value of total number of images).
    * @return new composite image.
    */
cv::Mat MakeCanvas(std::vector<cv::Mat> &vecMat, int windowHeight, int nRows) {
  int N = vecMat.size();
  nRows = nRows > N ? N : nRows;
  int edgeThickness = 10;
  int imagesPerRow = ceil(double(N) / nRows);
  int resizeHeight = floor(2.0 * ((floor(double(windowHeight - edgeThickness) / nRows)) / 2.0)) - edgeThickness;
  int maxRowLength = 0;

  std::vector<int> resizeWidth;
  for (int i = 0; i < N;) {
    int thisRowLen = 0;
    for (int k = 0; k < imagesPerRow; k++) {
      double aspectRatio = double(vecMat[i].cols) / vecMat[i].rows;
      int temp = int(ceil(resizeHeight * aspectRatio));
      resizeWidth.push_back(temp);
      thisRowLen += temp;
      if (++i == N) break;
    }
    if ((thisRowLen + edgeThickness * (imagesPerRow + 1)) > maxRowLength) {
      maxRowLength = thisRowLen + edgeThickness * (imagesPerRow + 1);
    }
  }
  int windowWidth = maxRowLength;
  cv::Mat canvasImage(windowHeight, windowWidth, CV_8UC3, cv::Scalar(0, 0, 0));

  for (int k = 0, i = 0; i < nRows; i++) {
    int y = i * resizeHeight + (i + 1) * edgeThickness;
    int x_end = edgeThickness;
    for (int j = 0; j < imagesPerRow && k < N; k++, j++) {
      int x = x_end;
      cv::Rect roi(x, y, resizeWidth[k], resizeHeight);
      cv::Size s = canvasImage(roi).size();
      // change the number of channels to three
      cv::Mat target_ROI(s, CV_8UC3);
      if (vecMat[k].channels() != canvasImage.channels()) {
        if (vecMat[k].channels() == 1) {
          cv::cvtColor(vecMat[k], target_ROI, CV_GRAY2BGR);
        }
      } else {
        vecMat[k].copyTo(target_ROI);
      }
      cv::resize(target_ROI, target_ROI, s);
      if (target_ROI.type() != canvasImage.type()) {
        target_ROI.convertTo(target_ROI, canvasImage.type());
      }
      target_ROI.copyTo(canvasImage(roi));
      x_end += resizeWidth[k] + edgeThickness;
    }
  }
  return canvasImage;
}

} // namespace dsl::relocalization