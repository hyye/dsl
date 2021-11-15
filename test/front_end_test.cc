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
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "dsl_common.h"
#include "full_system/coarse_tracker.h"
#include "full_system/full_system.h"

#include "tool/cv_helper.h"
#include "tool/new_tsukuba_reader.h"

using namespace dsl;
using namespace std;

TEST(FrontEndTest, PixelSelectorTest) {
  NewTsukubaReader reader("/home/hyye/Downloads/NewTsukubaStereoDataset");
  reader.ReadImageAndDepth(0);
  SetGlobalCalib(reader.w, reader.h, reader.K, 0);
  CalibHessian HCalib;

  unique_ptr<FrameHessian> fh = make_unique<FrameHessian>();
  cv::Mat cv_imgf, cv_imgg;
  cvtColor(reader.color_img, cv_imgg, CV_BGR2GRAY);
  cv_imgg.convertTo(cv_imgf, CV_32FC1);
  ImageAndExposure img(cv_imgf.cols, cv_imgf.rows, 1);
  memcpy(img.image.data(), cv_imgf.data,
         sizeof(float) * cv_imgf.cols * cv_imgf.rows);
  fh->MakeImages(img.image.data(), &HCalib);

  std::vector<float> map_out(wG[0] * hG[0]);
  PixelSelector pixel_selector(wG[0], hG[0]);
  pixel_selector.MakeMaps(*fh, map_out, 1500);
  for (int x = 0; x < wG[0]; ++x) {
    for (int y = 0; y < hG[0]; ++y) {
      int idx = x + wG[0] * y;
      if (map_out[idx] != 0) {
        cv::Point2f corner(x, y);
        cv::circle(reader.color_img, corner, 1, cv::Scalar(0, 0, 255), -1,
                   cv::FILLED);
      }
    }
  }
  cv::namedWindow("vis", cv::WINDOW_NORMAL);
  cv::imshow("vis", reader.color_img);
  cv::resizeWindow("vis", 640, 480);
  //  cv::waitKey(0);
}

TEST(FrontEndTest, DistanceInitializerTest) {
  NewTsukubaReader reader("/home/hyye/Downloads/NewTsukubaStereoDataset");
  reader.ReadImageAndDepth(0);
  SetGlobalCalib(reader.w, reader.h, reader.K, 0);
  CalibHessian HCalib;

  unique_ptr<FrameHessian> fh = make_unique<FrameHessian>();
  FrameHessian* fh_ptr = fh.get();
  cv::Mat cv_imgf, cv_imgg;
  cvtColor(reader.color_img, cv_imgg, CV_BGR2GRAY);
  cv_imgg.convertTo(cv_imgf, CV_32FC1);
  ImageAndExposure img(cv_imgf.cols, cv_imgf.rows, 1);
  memcpy(img.image.data(), cv_imgf.data,
         sizeof(float) * cv_imgf.cols * cv_imgf.rows);
  fh->MakeImages(img.image.data(), &HCalib);

  DistanceInitializer initializer(wG[0], hG[0]);

  Mat33f K_inv = reader.K.inverse();
  std::vector<float> all_dist(wG[0] * hG[0]);
  for (int x = 0; x < wG[0]; ++x) {
    for (int y = 0; y < hG[0]; ++y) {
      int idx = x + wG[0] * y;
      Vec3f pix(x, y, 1);
      Vec3f point = K_inv * pix * reader.depth_img.at<float>(y, x);
      float dist = point.norm();
      all_dist[idx] = dist;
    }
  }

  initializer.SetFirstDistance(HCalib, fh, all_dist, SE3());
  initializer.TrackFrameDepth(*initializer.first_frame, SE3());
  LOG(INFO) << initializer.points[0].size();

  for (auto&& pnt : initializer.points[0]) {
    cv::Point2f corner(pnt.u, pnt.v);
    cv::circle(reader.color_img, corner, 1, cv::Scalar(0, 0, 255), -1,
               cv::FILLED);
  }
  cv::namedWindow("vis", cv::WINDOW_NORMAL);
  cv::imshow("vis", reader.color_img);
  cv::resizeWindow("vis", 640, 480);
  //  cv::waitKey(0);
}

TEST(FrontEndTest, FirstSeveralFramesTest) {
  NewTsukubaReader reader("/home/hyye/Downloads/NewTsukubaStereoDataset");
  reader.ReadImageAndDepth(0);
  SetGlobalCalib(reader.w, reader.h, reader.K, 0);

  cv::Mat cv_imgf, cv_imgg;
  cvtColor(reader.color_img, cv_imgg, CV_BGR2GRAY);
  cv_imgg.convertTo(cv_imgf, CV_32FC1);
  ImageAndExposure img(cv_imgf.cols, cv_imgf.rows, 1);
  memcpy(img.image.data(), cv_imgf.data,
         sizeof(float) * cv_imgf.cols * cv_imgf.rows);

  FullSystem full_system;
  Mat33f K_inv = reader.K.inverse();
  std::vector<float> all_dist(wG[0] * hG[0]);
  for (int x = 0; x < wG[0]; ++x) {
    for (int y = 0; y < hG[0]; ++y) {
      int idx = x + wG[0] * y;
      Vec3f pix(x, y, 1);
      Vec3f point = K_inv * pix * reader.depth_img.at<float>(y, x);
      float dist = point.norm();
      all_dist[idx] = dist;
    }
  }
  full_system.AddActiveFrame(img, 0, all_dist, SE3());
  full_system.AddActiveFrame(img, 1);

  const DistanceInitializer& initializer = *full_system.distance_initializer;

  for (auto&& pnt : full_system.frame_hessians.front()->point_hessians) {
    cv::Point2f corner(pnt->u, pnt->v);
    cv::circle(reader.color_img, corner, 1, cv::Scalar(0, 0, 255), -1,
               cv::FILLED);
  }

  //  cv::namedWindow("vis", cv::WINDOW_NORMAL);
  //  cv::imshow("vis", reader.color_image);
  //  cv::resizeWindow("vis", 640, 480);
  //  cv::waitKey(0);

  reader.ReadImageAndDepth(22);  // MATLAB 5
  cvtColor(reader.color_img, cv_imgg, CV_BGR2GRAY);
  cv_imgg.convertTo(cv_imgf, CV_32FC1);
  memcpy(img.image.data(), cv_imgf.data,
         sizeof(float) * cv_imgf.cols * cv_imgf.rows);
  full_system.AddActiveFrame(img, 2);
  full_system.AddActiveFrame(img, 3);
  full_system.AddActiveFrame(img, 4);
  LOG(INFO) << "cam to world:" << std::endl
            << full_system.all_frame_shells.back()->cam_to_world.matrix3x4();
  LOG(INFO) << "gt:" << std::endl
            << reader.cam_pose.matrix3x4();
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_alsologtostderr = true;

  testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}