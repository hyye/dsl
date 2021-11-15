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
// Created by hyye on 12/27/19.
//

#include "map_render/core/global_model.h"
#include "map_render/core/index_map.h"
#include "map_render/data_loader.h"
#include "map_render/gui.h"
#include "map_render/yaml_loader.h"
#include "tool/euroc_reader.h"

using namespace dsl;

DEFINE_string(path, "", "input path");

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_alsologtostderr = true;

  EurocReader reader(FLAGS_path);
  YamlLoader yaml_loader(reader.config_path);
  MapLoader map_loader(yaml_loader.pcd_file);
  SetGlobalCalib(reader.w, reader.h, reader.K, 0);

  GUI gui(false, false);

  pangolin::GlBuffer *vbo = map_loader.GetVbo();
  LOG(INFO) << vbo->SizeBytes() << " size bytes";

  GlobalModel global_model;
  global_model.Initialization(*vbo);

  PangolinUtils new_vn;
  IndexMap index_map;
  Eigen::Affine3f trans;
  trans = yaml_loader.initial_pose;

  while (!pangolin::ShouldQuit()) {
    gui.PreCall();

    index_map.PredictGlobalIndices(trans.matrix(), global_model.Model(), 100);
    new_vn.SetVertexNormal(index_map.VertGlobalTex()->texture,
                           index_map.NormalGlobalTex()->texture);
    Vec4f *n_ptr = (Vec4f *)new_vn.normal_img.ptr;
    Vec4f *v_ptr = (Vec4f *)new_vn.vertex_img.ptr;
    cv::Mat normal_img(hG[0], wG[0], CV_32FC3);
    cv::Mat vertex_img(hG[0], wG[0], CV_32FC3);
    for (int x = 0; x < wG[0]; ++x) {
      for (int y = 0; y <  hG[0]; ++y) {
        int idx = x + y * wG[0];
        Vec4f n = n_ptr[idx];
        Vec4f v = v_ptr[idx];
        normal_img.at<cv::Vec3f>(y, x) = cv::Vec3f(n.x(), n.y(), n.z());
        vertex_img.at<cv::Vec3f>(y, x) = cv::Vec3f(v.x(), v.y(), v.z());
      }

    }

    cv::namedWindow("n", cv::WINDOW_NORMAL);
    cv::imshow("n", normal_img);

    cv::namedWindow("v", cv::WINDOW_NORMAL);
    cv::imshow("v", vertex_img / 10.0);
    cv::waitKey(1);

    if (gui.draw_global_model->Get()) {
      global_model.RenderPointCloud(gui.s_cam.GetProjectionModelViewMatrix(),
                                    gui.draw_normals->Get(),
                                    gui.draw_colors->Get());
    }

    gui.PostCall();
  }

  delete vbo;

  return 0;
}