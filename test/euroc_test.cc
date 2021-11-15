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
// Created by hyye on 12/19/19.
//

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <sophus/se3.hpp>
#include "full_system/full_system.h"
#include "tool/euroc_reader.h"

using namespace dsl;
using namespace std;

DEFINE_string(path, "", "input path");

std::string ToStr(std::string time, const SE3& se3) {
  std::stringstream ss;
  Eigen::Quaterniond q = se3.unit_quaternion();
  Vec3 p = se3.translation();
  ss << time.substr(0, 10) << "." << time.substr(10, 9) << " " << p.x() << " "
     << p.y() << " " << p.z() << " " << q.x() << " " << q.y() << " " << q.z()
     << " " << q.w() << std::endl;
  return ss.str();
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_alsologtostderr = true;

  EurocReader reader(FLAGS_path);

  cout << reader.K << endl;
  cout << reader.w << endl;
  cout << reader.h << endl;
  int id = 75;

  reader.ReadImageAndDist(id);
  // cv::namedWindow("g", CV_WINDOW_AUTOSIZE);
  // cv::imshow("g", reader.gray_image);
  // cv::namedWindow("d", CV_WINDOW_AUTOSIZE);
  // cv::imshow("d", reader.dist_image / 10.0);
  // cv::waitKey(0);

  SetGlobalCalib(reader.w, reader.h, reader.K, 0);
  cv::Mat cv_imgf;
  reader.gray_image.convertTo(cv_imgf, CV_32FC1);
  ImageAndExposure img(cv_imgf.cols, cv_imgf.rows, 1);
  memcpy(img.image.data(), cv_imgf.data,
         sizeof(float) * cv_imgf.cols * cv_imgf.rows);

  Mat33f K_inv = reader.K.inverse();
  std::vector<float> all_dist(wG[0] * hG[0]);
  for (int x = 0; x < wG[0]; ++x) {
    for (int y = 0; y < hG[0]; ++y) {
      int idx = x + wG[0] * y;
      Vec3f pix(x, y, 1);
      Vec3f point = K_inv * pix * reader.dist_image.at<float>(y, x);
      float dist = point.norm();
      all_dist[idx] = dist;
    }
  }

  FullSystem full_system;
  std::ofstream odom_file;
  odom_file.open("/tmp/euroc_odom.tum");

  full_system.AddActiveFrame(img, id, all_dist, SE3());
  odom_file << ToStr(reader.filenames[id],
                     full_system.all_frame_shells.back()->cam_to_world);

  for (int idx = 76; idx < reader.filenames.size(); ++idx) {
    reader.ReadImage(idx);
    reader.gray_image.convertTo(cv_imgf, CV_32FC1);

    memcpy(img.image.data(), cv_imgf.data,
           sizeof(float) * cv_imgf.cols * cv_imgf.rows);

    LOG(ERROR) << "idx: " << idx;
    full_system.AddActiveFrame(img, ++id);
    odom_file << ToStr(reader.filenames[id],
                       full_system.all_frame_shells.back()->cam_to_world);

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

  return 0;
}
