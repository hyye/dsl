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
// Created by hyye on 11/21/19.
//

//
// Created by hyye on 11/17/19.
//

#include <ceres/ceres.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <sophus/se3.hpp>
#include "full_system/full_system.h"
#include "optimization/photo_cost_functor.h"
#include "optimization/se3_local_parameterization.h"
#include "util/timing.h"

#include "tool/new_tsukuba_reader.h"

using namespace dsl;
using std::cout;
using std::endl;

typedef Eigen::Matrix<double, 6, 1> Vec6;
typedef Eigen::Matrix<double, 3, 1> Vec3;
typedef Sophus::SE3d SE3d;

TEST(CeresTest, HomoCostFunctorTest) {
  NewTsukubaReader reader("/home/hyye/Downloads/NewTsukubaStereoDataset");
  reader.ReadImageAndDepth(0);
  SetGlobalCalib(reader.w, reader.h, reader.K, 0);

  cv::Mat cv_imgf, cv_imgg;
  cvtColor(reader.color_img, cv_imgg, CV_BGR2GRAY);
  cv_imgg.convertTo(cv_imgf, CV_32FC1);
  ImageAndExposure img(cv_imgf.cols, cv_imgf.rows, 1);
  memcpy(img.image.data(), cv_imgf.data,
         sizeof(float) * cv_imgf.cols * cv_imgf.rows);

  // settingAffineOptModeA = -1;
  // settingAffineOptModeB = -1;

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
  int id = 0;

  std::ofstream gt_file, odom_file;
  gt_file.open("/tmp/nt_gt.tum");
  odom_file.open("/tmp/nt_odom.tum");

  auto ToStr = [](double time, const SE3& se3) {
    std::stringstream ss;
    Eigen::Quaterniond q = se3.unit_quaternion();
    Vec3 p = se3.translation();
    ss << time << " " << p.x() << " " << p.y() << " " << p.z() << " " << q.x()
       << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
    return ss.str();
  };

  full_system.AddActiveFrame(img, id, all_dist, SE3());
  gt_file << ToStr(id, reader.cam_pose);
  odom_file << ToStr(id, full_system.all_frame_shells.back()->cam_to_world);

  for (int idx = 1; idx < 1800; ++idx) {
    reader.ReadImageAndDepth(idx);  // MATLAB 5->22 7->30

    // cv::namedWindow("vis", cv::WINDOW_NORMAL);
    // cv::imshow("vis", reader.color_image);
    // cv::waitKey(1);
    cvtColor(reader.color_img, cv_imgg, CV_BGR2GRAY);
    cv_imgg.convertTo(cv_imgf, CV_32FC1);
    memcpy(img.image.data(), cv_imgf.data,
           sizeof(float) * cv_imgf.cols * cv_imgf.rows);
    LOG(ERROR) << "idx: " << idx;
    full_system.AddActiveFrame(img, ++id);
    gt_file << ToStr(id, reader.cam_pose);
    odom_file << ToStr(id, full_system.all_frame_shells.back()->cam_to_world);

    /*
    {
      FrameHessian* last_fh_ptr = full_system.frame_hessians.back().get();
      cv::Mat show_img(hG[0], wG[0], CV_8UC3);
      for (int x = 0; x < wG[0]; ++x) {
        for (int y = 0; y < hG[0]; ++y) {
          float c = last_fh_ptr->dI[x + y * wG[0]].x();
          show_img.at<cv::Vec3b>(y, x) = cv::Vec3b(c, c, c);
        }
      }
      for (auto&& fh : full_system.frame_hessians) {
        Vec3b color_idist(-1, -1, -1);
        float u, v;
        for (auto&& ph : fh->point_hessians) {
          u = -10, v = -10;
          if (fh.get() == last_fh_ptr) {
            color_idist = MakeJet3B(ph->idist);
            u = ph->u + 0.5;
            v = ph->v + 0.5;
          } else {
            for (auto&& r : ph->residuals) {
              if (r->target == last_fh_ptr) {
                auto& pp = r->center_projected_to;
                color_idist = MakeJet3B(pp.z());
                u = pp.x() + 0.5;
                v = pp.y() + 0.5;
              } else {
                continue;
              }
            }
          }
          cv::drawMarker(
              show_img, cv::Point(u, v),
              cv::Scalar(color_idist[2], color_idist[1], color_idist[0]),
              cv::MARKER_CROSS, 5, 1);
        }
      }
      cv::putText(show_img, std::to_string(idx), cv::Point(10, 50),
                  cv::FONT_HERSHEY_DUPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
      cv::namedWindow("vis", cv::WINDOW_NORMAL);
      cv::imshow("vis", show_img);
      cv::waitKey(1);
    }
    */
  }

  LOG(INFO)
      << full_system.frame_hessians.back()->shell->cam_to_world.matrix3x4();

  /*
  Eigen::IOFormat fmt(Eigen::StreamPrecision, Eigen::DontAlignCols, "; ", ", ",
                      "", "", "[", "]");
  for (const SE3& pose : reader.all_poses) {
    cout << pose.translation().transpose().format(fmt) << "# "
         << pose.rotationMatrix().format(fmt) << endl;
  }
  */

  /*
  {
    Vec6 tan_t, tan_h;
    Vec2 ab_t, ab_h;
    // tan_th.setZero();
    tan_t =
        full_system.frame_hessians.back()->shell->cam_to_ref.inverse().log();
    tan_h.setZero();
    ab_t = full_system.frame_hessians.back()->shell->aff_light.Vec();
    ab_h = full_system.frame_hessians.front()->shell->aff_light.Vec();
    std::vector<double> idists;
    ceres::Problem problem;
    idists.resize(full_system.active_residuals.size());

    ceres::LocalParameterization* local_parameterization =
        new SE3LocalParameterization();
    problem.AddParameterBlock(tan_t.data(), 6, local_parameterization);
    problem.AddParameterBlock(ab_t.data(), 2);
    problem.AddParameterBlock(tan_h.data(), 6, local_parameterization);
    problem.AddParameterBlock(ab_h.data(), 2);
    problem.SetParameterBlockConstant(tan_h.data());
    problem.SetParameterBlockConstant(ab_h.data());

    FrameHessian* fh_end = full_system.frame_hessians.back().get();
    for (int i = 0; i < full_system.active_residuals.size(); ++i) {
      auto&& res = full_system.active_residuals[i];
      if (res->target == fh_end) {
        idists[i] = full_system.active_residuals[i]->point->idist;
        // problem.AddParameterBlock(&idists[i], 1);
        ceres::CostFunction* cost_function = new PhotoCostFunctor(
            full_system.active_residuals[i], full_system.HCalib, true);
        problem.AddResidualBlockSpec(cost_function, NULL, tan_t.data(),
        ab_t.data(),
                                 tan_h.data(), ab_h.data(), &idists[i]);
      }
    }
    SE3d dummy_se3 = SE3d::exp(tan_t);
    // dummy_se3.setRotationMatrix(Mat33::Identity());
    // dummy_se3.translation() = Vec3(0, 0, -0.01);
    LOG(INFO) << "gt: " << endl << SE3d::exp(tan_t).matrix3x4();
    LOG(INFO) << "dummy: " << endl << dummy_se3.matrix3x4();
    tan_t = dummy_se3.log();

    LOG(INFO) << "initial" << endl << tan_t.transpose();
    LOG(INFO) << "ab initial" << endl << ab_t.transpose();

    // Run the solver!
    ceres::Solver::Options options;
    options.num_threads = NUM_THREADS;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    // options.max_solver_time_in_seconds = 0.1;
    // options.trust_region_strategy_type = ceres::DOGLEG;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);
    LOG(INFO) << summary.FullReport() << "\n";
    LOG(INFO) << "optimized" << endl << tan_t.transpose();
    LOG(INFO) << "optimized" << endl << SE3d::exp(tan_t).matrix3x4();
    LOG(INFO) << "ab " << ab_t.transpose();
  }
  */
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_alsologtostderr = true;

  testing::InitGoogleTest(&argc, argv);

  srand((unsigned int)time(0));

  return RUN_ALL_TESTS();
}
