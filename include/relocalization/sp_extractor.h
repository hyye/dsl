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
// Created by hyye on 8/8/20.
//

/**
 * @file sp_extractor.h
 * @author hyhuang hyhuang1995@gmail.com
 * @brief SuperPoint Extractor
 * @version 0.1
 * @date 2019-07-25
 *
 * @copyright Copyright (c) 2019
 *
 */

#ifndef DSL_SP_EXTRACTOR_H
#define DSL_SP_EXTRACTOR_H

#include <list>
#include <vector>

#include <Eigen/Dense>
#include <opencv/cv.h>

#include <torch/script.h>
#include <torch/torch.h>

#include "relocalization/feature_extractor.h"

namespace dsl::relocalization {

struct SPFrontend : torch::nn::Module {
  // public:
  SPFrontend(const float conf_thresh, const int height, const int width,
             const int cell_size);

  torch::Tensor grid;
  float conf_thresh;
  int h, w, hc, wc, c;

  // torch::Device device_;

  std::vector<torch::Tensor> forward(torch::Tensor x);

  torch::nn::Conv2d conv1a;
  torch::nn::Conv2d conv1b;

  torch::nn::Conv2d conv2a;
  torch::nn::Conv2d conv2b;

  torch::nn::Conv2d conv3a;
  torch::nn::Conv2d conv3b;

  torch::nn::Conv2d conv4a;
  torch::nn::Conv2d conv4b;

  torch::nn::Conv2d convPa;
  torch::nn::Conv2d convPb;

  // descriptor
  torch::nn::Conv2d convDa;
  torch::nn::Conv2d convDb;
};

class SPExtractor : public ORBextractor {
 public:
  enum PyTorchDevice {
    CPU = 0,
    CUDA = 1
  };

  SPExtractor(int nfeatures, std::string weight_path, PyTorchDevice device_type = CUDA, float conf_thresh = 0.007);

  virtual ~SPExtractor() = default;

  void operator()(cv::InputArray image, cv::InputArray mask,
                  std::vector<cv::KeyPoint> &keypoints,
                  cv::OutputArray descriptors) override;

  void nms(const cv::Mat &det, const cv::Mat &desc,
           std::vector<cv::KeyPoint> &pts, cv::Mat &descriptors, int border,
           int dist_thresh, int img_width, int img_height, cv::Mat &occ_grid);

  cv::Mat getMask() { return mask_; }

  cv::Mat getHeatMap() { return heat_; }

  const std::vector<Eigen::Vector2f> getCov() { return cov2_; }

  const std::vector<Eigen::Vector2f> getCov2Inv() { return cov2_inv_; }

  cv::Mat semi_dust_, dense_dust_;

  cv::Mat mask_, heat_, heat_inv_;

  cv::Mat occ_grid_;

 protected:
  PyTorchDevice device_type_;
  int num_features_;
  std::string weight_path_;
  std::vector<Eigen::Vector2f> cov2_, cov2_inv_;
  std::vector<Eigen::Matrix2f> info_mat_;
  // std::vector<cv::Point2f> cov2_inv_;

  // std::shared_ptr<torch::jit::script::Module> model_;
  torch::Device device_;

  float conf_thresh_ = 0.007;

  std::shared_ptr<SPFrontend> model_;
};

} // namespace

#endif // DSL_SP_EXTRACTOR_H
