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
// Created by hyye on 8/10/20.
//

#include <pangolin/pangolin.h>
#include <pangolin/gl/gl.h>
#include <gflags/gflags.h>

#include "relocalization/desc_map.h"
#include "relocalization/relocalization_config.h"
#include "relocalization/converter.h"
#include "map_render/map_render.h"

DEFINE_string(yaml_path,
              "/mnt/HDD/Datasets/Visual/EuRoC/V2_03_difficult/mav0/processed_data/relocalization/config.yaml",
              "path to yaml map file");
DEFINE_bool(run_opt, true, "run opt");

namespace visualization {

class MapVisualizer : public dsl::GUI {
 public:
  pangolin::Var<bool> *show_frames, *show_points, *show_covis;
  pangolin::Var<int> *selected_frame;
  MapVisualizer() {
    show_frames = new pangolin::Var<bool>("ui.show_frame", true, true);
    show_points = new pangolin::Var<bool>("ui.show_points", true, true);
    show_covis = new pangolin::Var<bool>("ui.show_covis", true, true);
    selected_frame = new pangolin::Var<int>("ui.selected_frame", 0, 0, 0);
  }
  ~MapVisualizer() {
    delete show_frames;
    delete show_points;
  }
};
}

using namespace dsl::relocalization;
using namespace visualization;

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_alsologtostderr = true;
  RelocalizationConfig yaml_loader(FLAGS_yaml_path);

  DescMap desc_map;
  desc_map.LoadMap(yaml_loader.database_map_path);

  dsl::Mat33f K;
  K << yaml_loader.gamma1, 0, yaml_loader.u0, 0, yaml_loader.gamma2,
      yaml_loader.v0, 0, 0, 1;
  dsl::Mat33f K_inv = K.inverse();

  dsl::SetGlobalCalib(yaml_loader.image_width, yaml_loader.image_height, K, yaml_loader.xi);

  std::map<unsigned long, Frame *> all_frames;
  for (auto const &[id, frame] :desc_map.all_keyframes)
    all_frames[id] = frame.get();
  std::map<unsigned long, MapPoint *> all_points;
  for (auto const &[id, point] :desc_map.all_map_points)
    all_points[id] = point.get();
  LOG(INFO) << "frames: " << all_frames.size();
  LOG(INFO) << "points: " << all_points.size();

  std::map<int, unsigned long> idx_to_db_id;
  int idx = 0;
  for (auto const &[id, frame] : all_frames) {
    LOG_ASSERT(id == frame->mnId);
    idx_to_db_id[idx] = id;
    idx += 1;
  }

  dsl::SE3 curr_pose = Converter::toSE3Quat(all_frames.begin()->second->GetPoseInverse());

  MapVisualizer gui;
  gui.FollowAbsPose(curr_pose.matrix().cast<float>());
  gui.selected_frame->Meta().range[1] = all_frames.size() - 1;

  dsl::MapLoader map_loader(yaml_loader.pcd_file, 1,
                            yaml_loader.intensity_scalar, yaml_loader.rgb_scalar, yaml_loader.surfel_size);
  pangolin::GlBuffer *vbo = map_loader.GetVbo();
  LOG(INFO) << vbo->SizeBytes() << " size bytes";

  dsl::GlobalModel global_model;
  global_model.Initialization(*vbo);

  desc_map.SetGlobalVertices(map_loader.vertices, map_loader.vertex_size, map_loader.ptc_size);

  typedef typename std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > vVec3f;

  vVec3f visible_pos;

  while (!pangolin::ShouldQuit()) {
    gui.PreCall();
    if (pangolin::Pushed(*gui.step) && *gui.selected_frame < all_frames.size() - 1) {
      *gui.selected_frame = *gui.selected_frame + 1;
    }

    if (gui.show_frames->Get()) {
      for (auto const &[id, frame] : all_frames) {
        dsl::SE3 frame_pose = Converter::toSE3Quat(frame->GetPoseInverse());
        glColor3f(1, 1, 0);
        gui.DrawFrustum(frame_pose.matrix().cast<float>());
        glColor3f(1, 1, 1);
      }

      visible_pos.clear();
      glLineWidth(3);
      int selected_frame = *gui.selected_frame;
      LOG_ASSERT(idx_to_db_id.count(selected_frame)) << selected_frame;
      unsigned long db_id = idx_to_db_id[selected_frame];
      Frame *pKF = all_frames[db_id];

      for (auto &&pMP : pKF->mvpMapPoints) {
        if (pMP && !pMP->isBad()) {
          Eigen::Vector3d pos = Converter::toVector3d(pMP->GetWorldPos());
          visible_pos.push_back(pos.cast<float>());
        }
      }
      std::vector<Frame *> vpCovisF = pKF->GetBestCovisibilityKeyFrames(5);
      curr_pose = Converter::toSE3Quat(pKF->GetPoseInverse());
      glColor3f(1, 0, 0);
      gui.DrawFrustum(curr_pose.matrix().cast<float>());
      glColor3f(1, 1, 1);
      for (auto &&pF : vpCovisF) {
        dsl::SE3 covis_pose = Converter::toSE3Quat(pF->GetPoseInverse());
        glColor3f(0, 1, 0);
        gui.DrawFrustum(covis_pose.matrix().cast<float>());
        glColor3f(1, 1, 1);

        for (auto &&pMP : pF->mvpMapPoints) {
          if (pMP && !pMP->isBad()) {
            Eigen::Vector3d pos = Converter::toVector3d(pMP->GetWorldPos());
            visible_pos.push_back(pos.cast<float>());
          }
        }
      }

    }

    if (gui.show_covis->Get()) {
      glLineWidth(1);
      vVec3f lines;
      int selected_frame = *gui.selected_frame;
      LOG_ASSERT(idx_to_db_id.count(selected_frame)) << selected_frame;
      unsigned long db_id = idx_to_db_id[selected_frame];
      for (auto const &[id, frame] : all_frames) {
        if (id != db_id) continue;
        dsl::SE3 frame_pose = Converter::toSE3Quat(frame->GetPoseInverse());
        for (Frame *pcF : frame->GetConnectedKeyFrames()) {
          dsl::SE3 cframe_pose = Converter::toSE3Quat(pcF->GetPoseInverse());
          lines.emplace_back(frame_pose.translation().cast<float>());
          lines.emplace_back(cframe_pose.translation().cast<float>());
        }
      }
      glColor3f(0, 1, 1);
      pangolin::glDrawLines(lines);
      glColor3f(1, 1, 1);
    }

    if (gui.draw_global_model->Get()) {
      global_model.RenderPointCloud(gui.s_cam.GetProjectionModelViewMatrix(),
                                    gui.draw_normals->Get(),
                                    gui.draw_colors->Get());
    }

    if (gui.followPose->Get()) {
      gui.FollowAbsPose(curr_pose.matrix().cast<float>());
    }
    if (gui.show_points->Get()) {
      vVec3f vec_pos;
      vVec3f normal_lines;
      if (all_points.size() != desc_map.all_map_points.size()) {
        all_points.clear();
        for (auto const &[id, point] :desc_map.all_map_points)
          all_points[id] = point.get();
        LOG(INFO) << "UPDATED: " << all_points.size();
      }
      for (auto const &[id, point] : all_points) {
        Eigen::Vector3d pos = Converter::toVector3d(point->GetWorldPos());
        vec_pos.push_back(pos.cast<float>());

        cv::Point3f normal = desc_map.GetGlobalNormal(point->idx_in_surfel_map);
        normal_lines.push_back(pos.cast<float>());
        normal_lines.push_back((pos + Eigen::Vector3d(normal.x, normal.y, normal.z) * 0.1).cast<float>());
      }
      gui.DrawWorldPoints(vec_pos, 2, Eigen::Vector3f(0, 1, 0));
      gui.DrawWorldPoints(visible_pos, 10, Eigen::Vector3f(1, 0, 0));

      if (gui.draw_normals->Get()) {
        glColor3f(1, 1, 1);
        pangolin::glDrawLines(normal_lines);
        glColor3f(1, 1, 1);
      }
    }

    if (pangolin::Pushed(*gui.save)) {
      timing::Timer save_timer("save");
      desc_map.SaveMap("/tmp/desc_map.bin");
      save_timer.Stop();

      {
        DescMap desc_map_tmp;
        desc_map_tmp.SetGlobalVertices(map_loader.vertices, map_loader.vertex_size, map_loader.ptc_size);
        timing::Timer load_timer("load");
        desc_map_tmp.LoadMap("/tmp/desc_map.bin");
        load_timer.Stop();

        // KeyFrameDatabase kf_db(&voc);
        // kf_db.BuildInvertedFile(desc_map_tmp.all_keyframes);

        LOG(INFO) << (VerifyDescMap(desc_map, desc_map_tmp) ? "Verified" : "Not verified");
      }
    }

    if (pangolin::Pushed(*gui.run_opt)) {
      LOG(INFO) << "RUN OPT";
      desc_map.DoOptimization(FLAGS_run_opt);
    }

    gui.PostCall();
  }

  return 0;
}