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
// Created by hyye on 11/5/19.
//

#include "dsl_common.h"
#include "full_system/full_system.h"
#include "tool/new_tsukuba_reader.h"
#include <gflags/gflags.h>
#include <gtest/gtest.h>

using namespace std;

using namespace dsl;

TEST(StructTest, RefTest) {
  unique_ptr<FullSystem> full_system = make_unique<FullSystem>();
  EnergyFunction *ef = full_system->ef.get();
  {
    unique_ptr<FrameHessian> last_fh = make_unique<FrameHessian>();
    unique_ptr<FrameHessian> new_fh = make_unique<FrameHessian>();
    unique_ptr<FrameShell> last_shell = std::make_unique<FrameShell>();
    unique_ptr<FrameShell> new_shell = std::make_unique<FrameShell>();
    last_fh->shell = last_shell.get();
    new_fh->shell = new_shell.get();
    last_fh->shell->id = 6;
    new_fh->shell->id = 7;

    CalibHessian HCalib;

    // TODO: is it weird?
    FrameHessian *last_fh_ptr = last_fh.get(); // &(*last_fh);
    FrameHessian *new_fh_ptr = new_fh.get();   // &(*new_fh);

    ef->InsertFrame(last_fh.get(), HCalib);
    ef->InsertFrame(new_fh.get(), HCalib);

    NewTsukubaReader reader("/home/hyye/Downloads/NewTsukubaStereoDataset");
    reader.ReadImageAndDepth(0);

    SetGlobalCalib(reader.w, reader.h, reader.K, 0);
    cv::Mat cv_imgf, cv_imgg;
    cvtColor(reader.color_img, cv_imgg, CV_BGR2GRAY);
    cv_imgg.convertTo(cv_imgf, CV_32FC1);
    ImageAndExposure img(cv_imgf.cols, cv_imgf.rows, 1);
    memcpy(img.image.data(), cv_imgf.data,
           sizeof(float) * cv_imgf.cols * cv_imgf.rows);
    last_fh_ptr->MakeImages(img.image.data(), &HCalib);

    ImmaturePoint pt(10, 10, last_fh_ptr, 1, HCalib);
    unique_ptr<PointHessian> last_ph = make_unique<PointHessian>(pt, HCalib);
    last_ph->pix = Vec2f(-1, -2);
    last_ph->host = last_fh_ptr;
    PointHessian *last_ph_ptr = last_ph.get();

    unique_ptr<PointFrameResidual> r =
        make_unique<PointFrameResidual>(last_ph_ptr, last_fh_ptr, new_fh_ptr);

    ef->InsertPoint(last_ph);
    ef->InsertResidual(r.get());
    ef->MakeIdx();
    r->point->residuals.emplace_back(std::move(r));

    full_system->frame_hessians.emplace_back(std::move(last_fh));
    full_system->frame_hessians.emplace_back(std::move(new_fh));
    full_system->all_frame_shells.emplace_back(std::move(last_shell));
    full_system->all_frame_shells.emplace_back(std::move(new_shell));
  }

  const FrameHessian &fh = *full_system->frame_hessians.front();
  const PointHessian &ph = *fh.point_hessians.front();
  cout << ph.pix << endl;
  cout << ph.host->shell->id << endl;
  cout << ph.residuals.front()->host->shell->id << endl;
  cout << ph.residuals.front()->target->shell->id << endl;
  cout << ph.residuals.front()->ef_residual->target->frame->shell->id << endl;
}

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_alsologtostderr = true;

  testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
