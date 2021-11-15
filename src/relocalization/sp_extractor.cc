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
 * @file sp_extractor.cpp
 * @author hyhuang hyhuang1995@gmail.com
 * @brief SuperPoint extractor
 * @version 0.1
 * @date 2019-07-25
 *
 * @copyright Copyright (c) 2019
 *
 */

#include <queue>
#include <random>

#include <opencv2/opencv.hpp>

#include "relocalization/sp_extractor.h"
#include "util/timing.h"
#include "util/global_calib.h"
#include "dsl_common.h"

using namespace cv;
using namespace std;

namespace dsl::relocalization {

const int c1 = 64;
const int c2 = 64;
const int c3 = 128;
const int c4 = 128;
const int c5 = 256;
const int d1 = 256;

SPFrontend::SPFrontend(const float conf_thresh_, const int height,
                       const int width, const int cell_size)
    : conf_thresh(conf_thresh_), h(height), w(width), hc(height / cell_size),
      wc(width / cell_size), c(cell_size),
      conv1a(torch::nn::Conv2dOptions(1, c1, 3).stride(1).padding(1)),
      conv1b(torch::nn::Conv2dOptions(c1, c1, 3).stride(1).padding(1)),

      conv2a(torch::nn::Conv2dOptions(c1, c2, 3).stride(1).padding(1)),
      conv2b(torch::nn::Conv2dOptions(c2, c2, 3).stride(1).padding(1)),

      conv3a(torch::nn::Conv2dOptions(c2, c3, 3).stride(1).padding(1)),
      conv3b(torch::nn::Conv2dOptions(c3, c3, 3).stride(1).padding(1)),

      conv4a(torch::nn::Conv2dOptions(c3, c4, 3).stride(1).padding(1)),
      conv4b(torch::nn::Conv2dOptions(c4, c4, 3).stride(1).padding(1)),

      convPa(torch::nn::Conv2dOptions(c4, c5, 3).stride(1).padding(1)),
      convPb(torch::nn::Conv2dOptions(c5, 65, 1).stride(1).padding(0)),

      convDa(torch::nn::Conv2dOptions(c4, c5, 3).stride(1).padding(1)),
      convDb(torch::nn::Conv2dOptions(c5, d1, 1).stride(1).padding(0)) {
  register_module("conv1a", conv1a);
  register_module("conv1b", conv1b);

  register_module("conv2a", conv2a);
  register_module("conv2b", conv2b);

  register_module("conv3a", conv3a);
  register_module("conv3b", conv3b);

  register_module("conv4a", conv4a);
  register_module("conv4b", conv4b);

  register_module("convPa", convPa);
  register_module("convPb", convPb);

  register_module("convDa", convDa);
  register_module("convDb", convDb);

  auto yx = torch::meshgrid({torch::arange(height), torch::arange(width)});
  // cout << x[0].sizes() << endl;
  // x[0].unsqueeze_(0)
  // x[1].unsqueeze_(0)
  auto grid_ = torch::cat({yx[1].unsqueeze(0), yx[0].unsqueeze(0)});
  grid = grid_.contiguous()
      .view({1, 2, height / 8, 8, width / 8, 8}) // TODO: batch-size
      .permute({0, 1, 3, 5, 2, 4})
      .reshape({1, 2, 64, hc, wc})
      .cuda();

  // torch::masked_select();
}

// -1 is okay for slicing
std::vector<torch::Tensor> SPFrontend::forward(torch::Tensor x) {

  x = torch::relu(conv1a->forward(x));
  x = torch::relu(conv1b->forward(x));
  x = torch::max_pool2d(x, 2, 2);

  x = torch::relu(conv2a->forward(x));
  x = torch::relu(conv2b->forward(x));
  x = torch::max_pool2d(x, 2, 2);

  x = torch::relu(conv3a->forward(x));
  x = torch::relu(conv3b->forward(x));
  x = torch::max_pool2d(x, 2, 2);

  x = torch::relu(conv4a->forward(x));
  x = torch::relu(conv4b->forward(x));

  auto cPa = torch::relu(convPa->forward(x));
  auto semi = convPb->forward(cPa).squeeze(); // [B, 65, H/8, W/8]

  auto cDa = torch::relu(convDa->forward(x));
  auto coarse = convDb->forward(cDa); // [B, d1, H/8, W/8]

  auto dn = torch::norm(coarse, 2, 1);
  // cout << "dn.sizes(): " << dn.sizes() << endl;
  coarse = coarse.div(torch::unsqueeze(dn, 1));
  // cout << "coarse.sizes(): " << coarse.sizes() << endl;

  auto dense = torch::softmax(semi, 0);
  auto semi_dust = semi[-1];
  auto dense_dust = dense[-1];
  auto nodust = dense.slice(0, 0, -1);
  // auto score, indices = ;

  // forward score and inlier indices
  auto score_idx = nodust.max(0);
  auto score = std::get<0>(score_idx);
  auto indices = std::get<1>(score_idx);
  // cout << score << endl << endl;
  // cout << indices << endl << endl;
  indices = indices.view({1, 1, 1, hc, wc}).expand({-1, 2, -1, -1, -1});
  auto pixel_grid = torch::gather(grid, 2, indices);
  auto pixel = torch::gather(grid, 2, indices);

  // auto mask = score >= 0.015;
  auto mask = score >= conf_thresh;
  auto pixels_in = torch::masked_select(pixel, mask);

  pixels_in = pixels_in.reshape({2, -1}).type_as(semi);
  score = torch::masked_select(score, mask);

  // forward heat
  auto heat_clamp = torch::clamp(nodust, 0.001);
  auto heat_log = torch::log(heat_clamp);
  auto heat_ = torch::pixel_shuffle(heat_log.unsqueeze(0), 8);
  // torch::log(heat_clamp);

  auto samp_pts = pixels_in.clone().cuda();
  auto x_s = samp_pts[0];
  auto y_s = samp_pts[1];
  x_s = x_s.div(w / 2.0) - 1.0;
  y_s = y_s.div(h / 2.0) - 1.0;
  // x_s = x_s / (w / 2.0) - 1.0;
  // y_s = y_s / (h / 2.0) - 1.0;

  samp_pts = torch::cat({x_s.unsqueeze(-1), y_s.unsqueeze(-1)}, -1);
  samp_pts = samp_pts.unsqueeze(0).unsqueeze(0);

  auto desc_sampled =
      torch::grid_sampler_2d(coarse, samp_pts, 0, 0, true).squeeze(2).squeeze();

  desc_sampled = desc_sampled.div(torch::norm(desc_sampled, 2, 0, true));

  std::vector<torch::Tensor> ret;
  ret.push_back(semi_dust);
  ret.push_back(dense_dust);
  ret.push_back(pixels_in);
  ret.push_back(score);
  ret.push_back(desc_sampled);
  ret.push_back(heat_);

  return ret;
}

void SPExtractor::nms(const cv::Mat &det, const cv::Mat &desc,
                      std::vector<cv::KeyPoint> &pts, cv::Mat &descriptors, int border,
                      int dist_thresh, int img_width, int img_height, cv::Mat &occ_grid) {

  std::vector<cv::Point2f> pts_raw;

  for (int i = 0; i < det.rows; i++) {

    int u = (int) det.at<float>(i, 0);
    int v = (int) det.at<float>(i, 1);
    // float conf = det.at<float>(i, 2);

    pts_raw.push_back(cv::Point2f(u, v));
  }

  cv::Mat grid = cv::Mat(cv::Size(img_width, img_height), CV_8UC1);
  cv::Mat inds = cv::Mat(cv::Size(img_width, img_height), CV_16UC1);
  occ_grid = cv::Mat(cv::Size(img_width / 8, img_height / 8), CV_16SC1, -1);

  grid.setTo(0);
  inds.setTo(0);

  for (int i = 0; i < pts_raw.size(); i++) {
    int uu = (int) pts_raw[i].x;
    int vv = (int) pts_raw[i].y;

    grid.at<char>(vv, uu) = 1;
    inds.at<unsigned short>(vv, uu) = i;
  }

  cv::copyMakeBorder(grid, grid, dist_thresh, dist_thresh, dist_thresh,
                     dist_thresh, cv::BORDER_CONSTANT, 0);

  int n_feature = 0;
  for (int i = 0; i < pts_raw.size(); i++) {
    int uu = (int) pts_raw[i].x + dist_thresh;
    int vv = (int) pts_raw[i].y + dist_thresh;

    if (grid.at<char>(vv, uu) != 1)
      continue;

    for (int k = -dist_thresh; k < (dist_thresh + 1); k++)
      for (int j = -dist_thresh; j < (dist_thresh + 1); j++) {
        if (j == 0 && k == 0)
          continue;

        grid.at<char>(vv + k, uu + j) = 0;
      }
    grid.at<char>(vv, uu) = 2;
    n_feature++;
    if (n_feature > num_features_) {
      break;
    }
  }

  size_t valid_cnt = 0;
  std::vector<int> select_indice;

  int16_t n_pts = 0;
  for (int v = 0; v < (img_height + dist_thresh); v++) {
    for (int u = 0; u < (img_width + dist_thresh); u++) {
      if (u - dist_thresh >= (img_width - border) || u - dist_thresh < border ||
          v - dist_thresh >= (img_height - border) || v - dist_thresh < border)
        continue;

      if (grid.at<char>(v, u) == 2) {
        occ_grid.at<int16_t>((v - dist_thresh) / 8, (u - dist_thresh) / 8) =
            n_pts++;
        int select_ind =
            (int) inds.at<unsigned short>(v - dist_thresh, u - dist_thresh);
        // WARNING: set to 31.0f
        pts.emplace_back(pts_raw[select_ind].x, pts_raw[select_ind].y, 31.0f);

        select_indice.push_back(select_ind);
        valid_cnt++;
      }
    }
  }

  descriptors = cv::Mat(select_indice.size(), 256, CV_32FC1);

  for (int i = 0; i < select_indice.size(); i++) {
    // for (int j = 0; j < 32; j++) {
    //   descriptors.at<unsigned char>(i, j) =
    //       desc.at<unsigned char>(select_indice[i], j);
    // }
    auto idx = select_indice[i];
    desc.row(idx).copyTo(descriptors.row(i));
  }
}

void computeCovariance(const cv::Mat &heat, vector<cv::KeyPoint> &kps,
                       vector<Eigen::Vector2f> &cov2,
                       vector<Eigen::Vector2f> &cov_inv,
                       vector<Eigen::Matrix2f> &info_mat) {
  const int h = heat.rows, w = heat.cols;
  cv::Mat viz = cv::Mat(h, w, CV_8UC3, cv::Scalar::all(255));
  cv::Mat occ_grid = cv::Mat(h, w, CV_8UC1, 1);

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist(0.0f, 255.0f);

  for (auto &kp : kps) {
    // const int u = kp.pt.x, v = kp.pt.y;
    vector<Eigen::Vector2f> delta;
    vector<float> score_patch;
    // delta.push_back(cv::Point2f(0.0f, 0.0f));

    const int uu = kp.pt.x, vv = kp.pt.y;
    kp.response = heat.at<float>(vv, uu);
    Eigen::Vector2f centre(uu, vv);

    queue<cv::Point2f> q;
    q.push(kp.pt);

    // heat = ;
    // auto color = cv::Vec3b(dist(mt), dist(mt), dist(mt));
    // Eigen::Vector<>

    while (!q.empty()) {
      auto &&pt = q.front();
      const int u = pt.x, v = pt.y;
      // viz.at<cv::Vec3b>(v, u) = color;
      occ_grid.at<uchar>(v, u) = 0;
      q.pop();

      Eigen::Vector2f delta_ = Eigen::Vector2f(u, v) - centre;
      delta.push_back(delta_.array().square());
      score_patch.push_back(heat.at<float>(v, u));

      float centroid = heat.at<float>(v, u);
      auto check_uv = [&](const int u_, const int v_) {
        float heat_value = heat.at<float>(v_, u_);
        if (occ_grid.at<uchar>(v_, u_) && heat_value > 0.0f &&
            heat_value < centroid) {
          q.push(Point2f(u_, v_));
        }
      };

      int xx, yy;
      xx = u - 1; // left
      if (xx > 0)
        check_uv(xx, v);
      yy = v - 1; // up
      if (yy > 0)
        check_uv(u, yy);
      xx = u + 1; // right
      if (xx < w)
        check_uv(xx, v);
      yy = v + 1; // down
      if (yy < h)
        check_uv(u, yy);
    }
    // if (score_patch.size() != delta.size())
    // if (score_patch.size() != kps.size())

    float sum_ = 0.0f;
    for (const auto &v : score_patch)
      sum_ += v;

    Eigen::Vector2f cov_ = Eigen::Vector2f::Zero(), cov_inv_;
    for (size_t i = 0; i < score_patch.size(); i++) {
      auto &delta_ = delta[i];
      cov_ += (score_patch[i] / sum_) * delta_;
    }
    if (cov_.x() < 1.0f)
      cov_.x() = 1.0f;
    if (cov_.y() < 1.0f)
      cov_.y() = 1.0f;

    cov_inv_ << 1.0f / cov_.x(), 1.0f / cov_.y();

    cov2.push_back(cov_);
    cov_inv.push_back(cov_inv_);

    // cout << cov_inv_.transpose() << endl;
  }

  // cv::imshow("patch", viz);
  // cv::waitKey(-1);
}

SPExtractor::SPExtractor(int nFeatures, std::string weight_path, SPExtractor::PyTorchDevice device_type, float conf_thresh)
    : ORBextractor(nFeatures, 1.2, 1, 20, 7),
      device_(torch::kCPU),
      num_features_(nFeatures),
      weight_path_(weight_path),
      device_type_(device_type),
      conf_thresh_(conf_thresh) {
  if (device_type_ == CUDA) {
    cout << "device set to CUDA" << endl;
    device_ = torch::Device(torch::kCUDA);
  }

  // model_ = torch::jit::load(sp::weight_path);
  model_ = make_shared<SPFrontend>(conf_thresh_, hG[0], wG[0], 8);
  torch::load(model_, weight_path_);
  model_->to(device_);
  model_->eval();
  // torch::load(model_, sp::weight_path);
}

void SPExtractor::operator()(InputArray _image, InputArray _mask,
                             vector<KeyPoint> &_keypoints,
                             OutputArray _descriptors) {
  LOG(WARNING) << "SP???";
  if (_image.empty())
    throw runtime_error("input image is empty");

  Mat image = _image.getMat();
  assert(image.type() == CV_8UC1);
  const auto height = image.rows;
  const auto width = image.cols;
  occ_grid_ = cv::Mat(height / 8, width / 8, CV_8UC1, cv::Scalar(0));

  // convert image to torch variable
  auto img_to_tensor = [](const cv::Mat &img, const torch::Device &device) {
    const auto height = img.rows;
    const auto width = img.cols;

    std::vector<int64_t> dims = {1, height, width, 1};
    auto img_var = torch::from_blob(img.data, dims, torch::kFloat32).to(device);
    img_var = img_var.permute({0, 3, 1, 2});
    img_var.set_requires_grad(false);

    return img_var;
  };

  cv::Mat im_raw, im_float;
  im_raw = _image.getMat();
  // cout << "" << endl;
  im_raw.convertTo(im_float, CV_32FC1, 1.f / 255.f, 0);

  auto img_var = img_to_tensor(im_float.clone(), device_);

  // std::vector<torch::jit::IValue> inputs = {img_var};
  // torch::cl

  std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
  timing::Timer timer_extration("sp extraction");
  // cout << "?" << endl;
  auto output = model_->forward(img_var);
  // cout << "?" << endl;
  std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
  timer_extration.Stop();
  double ttrack =
      std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
          .count();
  // cout << "extration time: " << ttrack << " ms" << endl;

  // .toTuple();

  // // auto semi = output->elements()[0].toTensor().squeeze();
  // // auto dense = output->elements()[1].toTensor().squeeze();
  // auto semi_dust = output->elements()[0].toTensor();
  // auto dense_dust = output->elements()[1].toTensor();
  // auto pts = output->elements()[2].toTensor();
  // auto score = output->elements()[3].toTensor().squeeze();
  // auto desc = output->elements()[4].toTensor().squeeze();
  // // auto sigma = output->elements()[3].toTensor();
  // auto heat = output->elements()[5].toTensor().squeeze().squeeze();
  // cout << sigma.sizes() << endl;
  // cout << heat.sizes() << endl;
  auto &semi_dust = output[0];
  auto &dense_dust = output[1];
  auto &pts = output[2];
  auto &score = output[3];
  auto &desc = output[4];
  // auto sigma = output->elements()[3].toTensor();
  auto &heat = output[5];

  // auto pts_cpu = pts.toType(torch::kFloat32).to(torch::kCPU);
  auto pts_cpu = pts.to(torch::kCPU);
  auto desc_cpu = desc.to(torch::kCPU);
  auto score_cpu = score.to(torch::kCPU);
  // auto sigma_cpu = sigma.to(torch::kCPU);
  auto heat_cpu = heat.to(torch::kCPU);
  auto dense_dust_cpu = dense_dust.to(torch::kCPU);
  auto semi_dust_cpu = semi_dust.to(torch::kCPU);

  int n_pts = pts_cpu.size(1);
  cv::Mat pts_mat(cv::Size(n_pts, 2), CV_32FC1, pts_cpu.data_ptr<float>());
  cv::Mat score_mat(
      cv::Size(n_pts, 1), CV_32FC1,
      score_cpu.data_ptr<float>()); // WARNING: Size(width, height)!!!!
  cv::Mat desc_mat(cv::Size(n_pts, 256), CV_32FC1, desc_cpu.data_ptr<float>());
  // cv::Mat sigma_mat(cv::Size(n_pts, 2), CV_32FC1,
  //                   sigma_cpu.data<float>());
  //                   height)!!!!
  cv::Mat heat_mat(cv::Size(width, height), CV_32FC1, heat_cpu.data_ptr<float>());

  // ADD: construct dustbin data
  semi_dust_ = cv::Mat(cv::Size(width / 8, height / 8), CV_32FC1,
                       semi_dust_cpu.data_ptr<float>());
  dense_dust_ = cv::Mat(cv::Size(width / 8, height / 8), CV_32FC1,
                        dense_dust_cpu.data_ptr<float>());

  // double min_, max_;
  // cv::minMaxLoc(dense_dust_, &min_, &max_);
  // cout << min_ << ' ' << max_ << endl;

  desc_mat = desc_mat.t();
  pts_mat = pts_mat.t();
  // sigma_mat = sigma_mat.t();

  auto to_heat = [](const cv::Mat &inp, cv::Mat &heat, cv::Mat &heat_inv) {
    cv::Mat out;
    auto img = inp * -1.0f;
    double min_, max_;

    cv::minMaxLoc(img, &min_, &max_);
    heat = (img - min_) / (max_ - min_);
    heat_inv = (max_ - img) / (max_ - min_);
    // img.convertTo(out, CV_8UC1, 255);
    // cv::applyColorMap(img.convertTo())
    // return img.;
    // return img;
  };
  to_heat(heat_mat, heat_, heat_inv_);
  // heat_ = to_heat(heat_mat);
  // cv::Mat out;
  // cv::Mat jet;
  // heat_.convertTo(out, CV_8UC1, 255);
  // cv::applyColorMap(out, jet, cv::COLORMAP_JET);
  // cv::imshow("viz", jet);
  // cv::waitKey(-1);

  //   keypoints.resize(n_pts);
  cv::Mat pts_sorted = pts_mat.clone();
  cv::Mat desc_sorted = desc_mat.clone();
  cv::Mat score_sorted = score_mat.clone();
  // cv::Mat sigma_sorted = sigma_mat.clone();

  cv::Mat indices;
  cv::sortIdx(score_mat, indices, cv::SORT_EVERY_ROW | cv::SORT_DESCENDING);
  for (int i = 0; i < indices.cols; i++) {
    auto idx = indices.at<int>(0, i);
    score_sorted.at<float>(0, i) = score_mat.at<float>(0, idx);
    pts_mat.row(idx).copyTo(pts_sorted.row(i));
    desc_mat.row(idx).copyTo(desc_sorted.row(i));
    // sigma_mat.row(idx).copyTo(sigma_sorted.row(i));
    // cout << desc_sorted
  }

  std::vector<cv::KeyPoint> kps;
  cv::Mat desc_;
  nms(pts_sorted, desc_sorted, kps, desc_, 8, 4, im_raw.cols, im_raw.rows,
      occ_grid_);

  cov2_.clear();
  cov2_inv_.clear();
  info_mat_.clear();
  computeCovariance(heat_inv_, kps, cov2_, cov2_inv_, info_mat_);

  // if (cov2_inv_.size() != kps.size()) {
  // }

  _keypoints = kps;

  _descriptors.create(kps.size(), 256, CV_32FC1);
  desc_.copyTo(_descriptors.getMat());
}

} // namespace