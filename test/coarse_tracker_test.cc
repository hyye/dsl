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

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "dsl_common.h"
#include "full_system/coarse_tracker.h"
#include "full_system/full_system.h"

#include "tool/cv_helper.h"
#include "tool/new_tsukuba_reader.h"

using namespace cv_helper;

using namespace dsl;

using namespace std;

TEST(CoarseTrackerTest, CoarseDistanceMapTest) {
  int i = 0;
  cout << i << endl, ++i, cout << i << endl;
  unique_ptr<FullSystem> full_system = make_unique<FullSystem>();
  unique_ptr<FrameHessian> last_fh = make_unique<FrameHessian>();
  unique_ptr<FrameHessian> new_fh = make_unique<FrameHessian>();

  CalibHessian HCalib;
  NewTsukubaReader reader("/home/hyye/Downloads/NewTsukubaStereoDataset");
  reader.ReadImageAndDepth(0);

  SetGlobalCalib(reader.w, reader.h, reader.K, 0);
  cv::Mat cv_imgf, cv_imgg;
  cvtColor(reader.color_img, cv_imgg, CV_BGR2GRAY);
  cv_imgg.convertTo(cv_imgf, CV_32FC1);
  ImageAndExposure img(cv_imgf.cols, cv_imgf.rows, 1);
  memcpy(img.image.data(), cv_imgf.data,
         sizeof(float) * cv_imgf.cols * cv_imgf.rows);
  last_fh->MakeImages(img.image.data(), &HCalib);

  ImmaturePoint pt(10, 10, last_fh.get(), 1, HCalib);
  unique_ptr<PointHessian> last_ph = make_unique<PointHessian>(pt, HCalib);
  last_ph->idist = 1;
  last_ph->p_sphere = Vec3f(0, 0, 1);
  last_fh->point_hessians.emplace_back(std::move(last_ph));

  full_system->frame_hessians.emplace_back(std::move(last_fh));
  full_system->frame_hessians.emplace_back(std::move(new_fh));

  Mat33f K;
  K << 320, 0, 320, 0, 320, 240, 0, 0, 1;
  SetGlobalCalib(640, 480, K, 0);

  CoarseDistanceMap coarse_distance_map(640, 480);
  coarse_distance_map.MakeK(HCalib);
  coarse_distance_map.MakeDistanceMap(full_system->frame_hessians,
                                      *full_system->frame_hessians.back());

  coarse_distance_map.AddIntoDistFinal(77, 77);

  cv::Mat img_f(240, 320, CV_32FC1,
                coarse_distance_map.fwd_warped_dist_final.data());
  cv::Mat img_g;
  img_f.convertTo(img_g, CV_8UC1);
  //  cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
  //  cv::imshow("Display window", img_g * 4);
  //
  //  cv::waitKey(0);
}

TEST(CoarseDistanceMapTest, PipelineTest) {
  NewTsukubaReader reader("/home/hyye/Downloads/NewTsukubaStereoDataset");
  reader.ReadImageAndDepth(0);
  /*
  cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
  cv::imshow("Display window", reader.color_image);
  cv::waitKey(0);
  cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
  cv::imshow("Display window", reader.depth_image);
  cv::waitKey(0);
   */
  SetGlobalCalib(reader.w, reader.h, reader.K, 0);

  cv::Mat color_ref, color_curr;

  unique_ptr<FullSystem> full_system = make_unique<FullSystem>();
  unique_ptr<FrameHessian> last_fh = make_unique<FrameHessian>();
  unique_ptr<FrameHessian> new_fh = make_unique<FrameHessian>();
  unique_ptr<FrameShell> last_shell = std::make_unique<FrameShell>();
  unique_ptr<FrameShell> new_shell = std::make_unique<FrameShell>();
  last_fh->shell = last_shell.get();
  new_fh->shell = new_shell.get();
  last_fh->shell->id = 0;
  new_fh->shell->id = 1;

  FrameHessian *last_fh_ptr = last_fh.get();
  FrameHessian *new_fh_ptr = new_fh.get();

  CalibHessian HCalib;
  CoarseTracker coarse_tracker(reader.w, reader.h);
  coarse_tracker.MakeK(HCalib);
  cv::Mat cv_imgf, cv_imgg;
  cvtColor(reader.color_img, cv_imgg, CV_BGR2GRAY);
  cv_imgg.convertTo(cv_imgf, CV_32FC1);
  color_ref = reader.color_img;
  ImageAndExposure img(cv_imgf.cols, cv_imgf.rows, 1);
  memcpy(img.image.data(), cv_imgf.data,
         sizeof(float) * cv_imgf.cols * cv_imgf.rows);
  last_fh_ptr->MakeImages(img.image.data(), &HCalib);
  new_fh_ptr->MakeImages(img.image.data(), &HCalib);

  full_system->frame_hessians.emplace_back(std::move(last_fh));
  full_system->frame_hessians.emplace_back(std::move(new_fh));
  full_system->all_frame_shells.emplace_back(std::move(last_shell));
  full_system->all_frame_shells.emplace_back(std::move(new_shell));

  full_system->ef->InsertFrame(last_fh_ptr, HCalib);
  full_system->ef->InsertFrame(new_fh_ptr, HCalib);
  Mat33f K_inv = reader.K.inverse();

  int img_size = reader.color_img.cols * reader.color_img.rows;

  vector<cv::Point2f> corners;
  /*
  for (int x = 20; x < reader.color_image.cols - 20;
       x += reader.color_image.cols / 40) {
    for (int y = 20; y < reader.color_image.rows - 20;
         y += reader.color_image.rows / 40) {
      corners.emplace_back(x, y);
    }
  }
  */

  /*
  {
    double qualityLevel = 0.01;
    double minDistance = 10;
    int blockSize = 3;
    bool useHarrisDetector = false;
    double k = 0.04;
    int maxCorners = 1000;
    goodFeaturesToTrack(cv_imgg, corners, maxCorners, qualityLevel, minDistance,
                        cv::Mat(), blockSize, useHarrisDetector, k);
    LOG(INFO) << "size: " << corners.size();
  }
  */

  std::vector<float> map_out(wG[0] * hG[0]);
  PixelSelector pixel_selector(wG[0], hG[0]);
  pixel_selector.MakeMaps(*full_system->frame_hessians.front(), map_out, 1000);
  for (int x = 0; x < wG[0]; ++x) {
    for (int y = 0; y < hG[0]; ++y) {
      int idx = x + wG[0] * y;
      if (map_out[idx] != 0) {
        corners.emplace_back(x, y);
      }
    }
  }
  LOG(INFO) << "size: " << corners.size();

  for (const cv::Point2f &corner : corners) {
    int x = corner.x;
    int y = corner.y;
    ImmaturePoint pt(x, y, last_fh_ptr, 1, HCalib);
    unique_ptr<PointHessian> ph_tmp = make_unique<PointHessian>(pt, HCalib);
    PointHessian *ph_ptr = ph_tmp.get();
    ph_tmp->host = last_fh_ptr;
    unique_ptr<PointFrameResidual> r =
        make_unique<PointFrameResidual>(ph_tmp.get(), last_fh_ptr, new_fh_ptr);
    PointFrameResidual *r_ptr = r.get();
    full_system->ef->InsertPoint(ph_tmp);
    full_system->ef->InsertResidual(r.get());
    r->point->residuals.emplace_back(std::move(r));
    ph_ptr->last_residuals[0].first = r_ptr;
    ph_ptr->last_residuals[0].second = ResState::IN;
    r_ptr->center_projected_to[0] = x;
    r_ptr->center_projected_to[1] = y;
    Vec3f pix(x, y, 1);
    Vec3f point = K_inv * pix * reader.depth_img.at<float>(y, x);
    r_ptr->center_projected_to[2] = 1.0 / point.norm();
    r_ptr->ef_residual->is_active_and_is_good_new = true;
    ph_ptr->convereged_ph_idist = true;
  }
  full_system->ef->MakeIdx();

  coarse_tracker.SetCoarseTrackingRef(full_system->frame_hessians);

  {
    unique_ptr<FrameHessian> new_fh = make_unique<FrameHessian>();
    unique_ptr<FrameShell> new_shell = std::make_unique<FrameShell>();
    new_fh->shell = new_shell.get();
    new_fh->shell->id = 2;

    FrameHessian *new_fh_ptr = new_fh.get();

    full_system->frame_hessians.emplace_back(std::move(new_fh));
    full_system->all_frame_shells.emplace_back(std::move(new_shell));

    reader.ReadImageAndDepth(22);  // MATLAB 5
    cv::Mat cv_imgf, cv_imgg;
    cvtColor(reader.color_img, cv_imgg, CV_BGR2GRAY);
    cv_imgg.convertTo(cv_imgf, CV_32FC1);
    ImageAndExposure img(cv_imgf.cols, cv_imgf.rows, 1);
    memcpy(img.image.data(), cv_imgf.data,
           sizeof(float) * cv_imgf.cols * cv_imgf.rows);
    new_fh_ptr->MakeImages(img.image.data(), &HCalib);
    LOG(INFO) << "cam_pose:" << std::endl << reader.cam_pose.matrix3x4();
    color_curr = reader.color_img;
  }

  SE3 last_to_new_out;
  AffLight aff_light_out;
  Vec5 achieve_res = Vec5::Constant(NAN);
  coarse_tracker.TrackNewestCoarse(*full_system->frame_hessians.back(),
                                   last_to_new_out, aff_light_out,
                                   pyrLevelsUsed - 1, achieve_res);
  LOG(INFO) << "last_to_new_out:" << std::endl
            << last_to_new_out.inverse().matrix3x4();
  LOG(INFO) << aff_light_out.Vec().transpose();

  vector<cv::Point2f> corners_curr;
  for (int i = 0; i < corners.size(); ++i) {
    corners_curr.emplace_back(coarse_tracker.buf_warped_u[i],
                              coarse_tracker.buf_warped_v[i]);
  }

  VisualizePairs(color_ref, color_curr, corners, corners_curr, false);
  //  cv::waitKey(0);

  /*
  for (int lvl = 0; lvl < pyrLevelsUsed; ++lvl) {
    LOG(INFO) << hG[lvl] << " " << wG[lvl] << " " << pyrLevelsUsed;
    cv::Mat idist_img =
        cv::Mat(hG[lvl], wG[lvl], CV_32FC1, coarse_tracker.idist[lvl].data());
    cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
    cv::imshow("Display window", idist_img);

    cv::waitKey(0);
  }
  */
}

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_alsologtostderr = true;

  testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}