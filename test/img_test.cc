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
// Created by hyye on 11/7/19.
//

#include "dsl_common.h"
#include "full_system/full_system.h"
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

using namespace dsl;

DEFINE_string(
    filename,
    "/home/hyye/Desktop/cyt/data_191005/images/1570262736750159414.jpg",
    "input image");

TEST(ImageTest, MinimalImageTest) {
  cv::Mat cv_img = cv::imread(FLAGS_filename);
  auto type2str = [](int type) {
    std::string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch (depth) {
    case CV_8U:
      r = "8U";
      break;
    case CV_8S:
      r = "8S";
      break;
    case CV_16U:
      r = "16U";
      break;
    case CV_16S:
      r = "16S";
      break;
    case CV_32S:
      r = "32S";
      break;
    case CV_32F:
      r = "32F";
      break;
    case CV_64F:
      r = "64F";
      break;
    default:
      r = "User";
      break;
    }

    r += "C";
    r += (chans + '0');

    return r;
  };
  LOG(INFO) << type2str(cv_img.type());
  MinimalImageB3 img(cv_img.cols, cv_img.rows, (Vec3b *)cv_img.data);
  LOG(INFO) << "img: " << (img.own_data.empty() ? "not own" : "own");
  std::unique_ptr<MinimalImageB3> img_clone = img.GetClone();
  LOG(INFO) << "img_clone: "
            << ((*img_clone).own_data.empty() ? "not own" : "own");

  cv::Mat cv_img_ref =
      cv::Mat(cv_img.rows, cv_img.cols, cv_img.type(), img_clone->data);

  //  cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
  //  cv::imshow("Display window", cv_img_ref);
  //
  //  cv::waitKey(1);
}

TEST(ImageTest, ImageAndExposureTest) {
  cv::Mat cv_img = cv::imread(FLAGS_filename, cv::IMREAD_GRAYSCALE);
  cv::Mat cv_imgf;
  cv_img.convertTo(cv_imgf, CV_32FC1);
  ImageAndExposure img(cv_imgf.cols, cv_imgf.rows, 1);
  memcpy(img.image.data(), cv_imgf.data,
         sizeof(float) * cv_imgf.cols * cv_imgf.rows);
  for (auto &&p : img.image) {
    p /= 255.0;
  }
  //  cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
  //  cv::imshow("Display window", cv::Mat(cv_imgf.rows, cv_imgf.cols,
  //                                       cv_imgf.type(), img.image.data()));
  //
  //  cv::waitKey(0);

  ImageAndExposure img_default(0, 0);
  {
    std::unique_ptr<ImageAndExposure> img_ptr =
        std::make_unique<ImageAndExposure>(img);
    //    cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
    //    cv::imshow("Display window",
    //               cv::Mat(cv_imgf.rows, cv_imgf.cols, cv_imgf.type(),
    //                       img_ptr->image.data()));
    //
    //    cv::waitKey(0);

    img_default = *img_ptr; // deep copy?

    LOG(INFO) << "img_ptr->image.data addr: " << (long)img_ptr->image.data();
  }

  LOG(INFO) << "img_default.image.data addr: "
            << (long)img_default.image.data();
  LOG(INFO) << "img.image.data addr: " << (long)img.image.data();

  //  cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
  //  cv::imshow("Display window",
  //             cv::Mat(cv_imgf.rows, cv_imgf.cols, cv_imgf.type(),
  //                     img_default.image.data()));
  //
  //  cv::waitKey(0);
}

TEST(ImageTest, InterpolateTest) {
  cv::Mat cv_img = cv::imread(FLAGS_filename, cv::IMREAD_GRAYSCALE);
  cv::Mat cv_imgf;
  cv_img.convertTo(cv_imgf, CV_32FC1);
  ImageAndExposure img(cv_imgf.cols, cv_imgf.rows, 1);
  memcpy(img.image.data(), cv_imgf.data,
         sizeof(float) * cv_imgf.cols * cv_imgf.rows);
  for (auto &&p : img.image) {
    p /= 255.0;
  }

  timing::Timer timer_set("test/set_global_calib");
  SetGlobalCalib(cv_imgf.cols, cv_imgf.rows, Mat33f(), 1);
  timer_set.Stop();

  std::unique_ptr<CalibHessian> HCalib = std::make_unique<CalibHessian>();
  FrameHessian fh;
  timing::Timer timer_make("test/make_image");
  fh.MakeImages(img.image.data(), HCalib.get());
  //  fh.MakeImages((float *)cv_imgf.data, HCalib.get());
  timer_make.Stop();

  int x = 400, y = 300, w = wG[0];
  float Ip = img.image[(int)x + ((int)y) * w];
  LOG(INFO) << Ip;
  timing::Timer timer_inter("test/interpolate");
  Vec3f hit_color = (GetInterpolatedElement33(fh.dI, x, y, w));
  timer_inter.Stop();
  LOG(INFO) << hit_color.transpose();

  LOG(INFO) << timing::Timing::Print();
  EXPECT_FLOAT_EQ(hit_color.x(), Ip);
}

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_alsologtostderr = true;

  testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}