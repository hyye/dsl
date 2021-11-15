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
// Created by hyye on 12/30/19.
//

#include "relocalization/sp_extractor.h"
#include "relocalization/sp_matcher.h"

#include "map_render/map_render.h"
#include "relocalization/feature_extractor.h"
#include "relocalization/feature_matcher.h"

#include "relocalization/vocabulary_binary.h"
#include "relocalization/desc_map.h"
#include "relocalization/struct/key_frame_database.h"
#include "relocalization/pnp_solver.h"
#include "relocalization/converter.h"
#include "tool/euroc_reader.h"
#include "tool/dataset_converter.h"
#include "util/timing.h"

#include "fmt/format.h"
#include "fmt/color.h"

using namespace dsl;
using namespace dsl::relocalization;

DEFINE_string(path,
              "/mnt/HDD/Datasets/Visual/EuRoC/V1_02_medium/mav0/processed_data/left_pinhole/relocalization",
              "input path");
DEFINE_string(traj_path,
              "/mnt/HDD/Datasets/Visual/EuRoC/V1_02_medium/mav0/processed_data/left_pinhole/relocalization/V102_left_kf.tum",
              "traj path");
DEFINE_string(voc_path, "/home/hyye/dev_ws/src/dsl/support_files/vocabulary/ORBvoc.bin", "vocabulary path");
DEFINE_bool(offscreen, false, "offscreen");
DEFINE_bool(pause, false, "pause");
DEFINE_bool(kf_culling, true, "kf_culling");
DEFINE_string(load, "", "Map to load");
DEFINE_bool(vis, true, "visualization");
DEFINE_bool(validate, true, "validation");
DEFINE_bool(run_opt, true, "run opt");

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_alsologtostderr = true;
  FLAGS_colorlogtostderr = true;

  OrbVocabularyBinary voc;
  voc.loadFromBinaryFile(FLAGS_voc_path);

  std::vector<FrameShellWithFn> all_frames_with_fn = LoadTum(FLAGS_traj_path);

  EurocReader reader(FLAGS_path);

  RelocalizationConfig yaml_loader(reader.config_path);
  LOG(INFO) << yaml_loader.Print();

  std::unique_ptr<ORBextractor>
      feature_extractor = std::make_unique<ORBextractor>(yaml_loader.max_num_features, 1.2, 8, 20, 7);

  MapLoader map_loader(yaml_loader.pcd_file, 1,
                       yaml_loader.intensity_scalar, yaml_loader.rgb_scalar, yaml_loader.surfel_size);
  Mat33f K;
  K << yaml_loader.gamma1, 0, yaml_loader.u0, 0, yaml_loader.gamma2,
      yaml_loader.v0, 0, 0, 1;
  Mat33f K_inv = K.inverse();
  // cv::Mat cv_K;
  // cv::eigen2cv(K, cv_K);

  SetGlobalCalib(yaml_loader.image_width, yaml_loader.image_height, K,
                 yaml_loader.xi);
  boost::filesystem::path mask_path = boost::filesystem::path(FLAGS_path) / "mask.png";
  if (boost::filesystem::exists(mask_path)) {
    SetGlobalMask(mask_path.string());
    LOG(WARNING) << "MASK LOADED: " << mask_path;
  }

  use_superpoint = yaml_loader.use_superpoint;
  if (use_superpoint) {
    SPExtractor::PyTorchDevice pytorch_device =
        yaml_loader.pytorch_device == "CUDA" ? SPExtractor::PyTorchDevice::CUDA : SPExtractor::PyTorchDevice::CPU;
    feature_extractor = std::make_unique<SPExtractor>(1000,
                                                      yaml_loader.superpoint_path,
                                                      pytorch_device,
                                                      yaml_loader.superpoint_conf_thresh);
  }
  LOG(INFO) << (use_superpoint ? "use sp" : "no sp");

  GUI gui(FLAGS_offscreen, false);
  float point_ratio = 0;
  gui.dataIdx->Meta().range[1] = all_frames_with_fn.size();
  gui.pause->operator=(FLAGS_pause);

  pangolin::GlBuffer *vbo = map_loader.GetVbo();
  LOG(INFO) << vbo->SizeBytes() << " size bytes";

  GlobalModel global_model;
  global_model.Initialization(*vbo);

  PangolinUtils new_vn;
  IndexMap index_map;
  Eigen::Affine3f kf_transform, ex_trans;
  ex_trans = yaml_loader.T_lc;
  kf_transform = yaml_loader.initial_pose * ex_trans;

  std::map<std::string, GPUTexture *> textures;
  textures[GPUTexture::RGB] = new GPUTexture(wG[0], hG[0], GL_RGBA, GL_RGB,
                                             GL_UNSIGNED_BYTE, true, true);
  textures["Model"] = new GPUTexture(wG[0], hG[0], GL_LUMINANCE32F_ARB, GL_LUMINANCE, GL_FLOAT);

  int id = 0;
  SE3 curr_pose = all_frames_with_fn.front().cam_to_world;
  SE3f pnp_pose;
  gui.FollowAbsPose(curr_pose.matrix().cast<float>());

  DescMap desc_map;
  desc_map.SetGlobalVertices(map_loader.vertices, map_loader.vertex_size, map_loader.ptc_size);
  desc_map.SetVocabularyBinary(&voc);

  if (boost::filesystem::exists(FLAGS_load)) {
    timing::Timer load_timer("load");
    desc_map.LoadMap(FLAGS_load);
    desc_map.SetFixMap(true);
  }

  desc_map.SetKeyFrameCulling(FLAGS_kf_culling);
  LOG(INFO) << (FLAGS_kf_culling ? "KF CULLING" : "KF NOT CULLING");

  typedef typename std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > vVec3f;
  vVec3f trajectories, relocalization_traj;
  vVec3f neighbor_points, old_neighbor_points;
  vVec3f associated_points;
  vVec3f desc_points;

  // TODO: move GUI to another thread
  while (!pangolin::ShouldQuit()) {
    gui.PreCall();

    if ((!gui.pause->Get() || pangolin::Pushed(*gui.step)) && id < all_frames_with_fn.size()) {

      const std::vector<std::string> &db_fns = reader.filenames;
      const FrameShellWithFn &fs_with_fn = all_frames_with_fn[id];
      int id_in_dataset =
          std::distance(db_fns.begin(), std::find(db_fns.begin(), db_fns.end(), fs_with_fn.filename));
      // LOG(INFO) << "@@@@@@@@ id: " << id << ", id_in_dataset: " << id_in_dataset << std::endl
      //           << all_frames_with_fn[id].filename << " " << db_fns[id_in_dataset];
      LOG_ASSERT(fs_with_fn.filename == db_fns[id_in_dataset]);

      reader.ReadImage(id_in_dataset);
      cv::Mat cv_imgf;
      reader.gray_image.convertTo(cv_imgf, CV_32FC1);

      curr_pose = fs_with_fn.cam_to_world;

      pangolin::TypedImage pgl_dist_img;
      index_map.SynthesizeDepth(curr_pose.matrix().cast<float>(), global_model.Model(), 100);
      index_map.DepthTex()->texture->Download(pgl_dist_img);
      cv::Mat dist_img_f = cv::Mat(hG[0], wG[0], CV_32FC1, pgl_dist_img.ptr);

      timing::Timer index_timer("index");
      std::vector<PointWithIndex> points_with_indices;

      {
        // index_global_img will help with the occlusion
        pangolin::TypedImage index_global_img, index_img;
        index_global_img.Base::Reinitialise(hG[0], wG[0], wG[0] * 32 / 8);
        index_img.Base::Reinitialise(hG[0], wG[0], wG[0] * 32 / 8);

        index_map.PredictGlobalIndices(curr_pose.matrix().cast<float>(), global_model.Model(), 100);
        index_map.PredictIndices(curr_pose.matrix().cast<float>(), global_model.Model(), 100);

        pangolin::GlTexture *tmp_texture = index_map.index_global_texture_.texture;
        tmp_texture->Bind();
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, index_global_img.ptr);
        tmp_texture->Unbind();

        tmp_texture = index_map.index_texture_.texture;
        tmp_texture->Bind();
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, index_img.ptr);
        tmp_texture->Unbind();

        // TODO: do averaging instead of picking single point?
        std::unordered_set<unsigned int> set_idx_already_added;
        for (int y = 0; y < hG[0]; ++y) {
          for (int x = 0; x < wG[0]; ++x) {
            unsigned int up_pt_idx =
                *(unsigned int *) &index_global_img.ptr[x * 4 + y * wG[0] * 32 / 8];
            unsigned int pt_idx =
                *(unsigned int *) &index_img.ptr[x * 4 + y * wG[0] * 32 / 8];
            float dist = dist_img_f.at<float>(cv::Point(x + 0.5, y + 0.5));
            if (pt_idx == 0 || dist == 0 || up_pt_idx == 0 /*|| pt_idx != up_pt_idx*/) continue;
            if (!set_idx_already_added.count(up_pt_idx)) {
              points_with_indices.emplace_back(x, y, up_pt_idx, dist);
              set_idx_already_added.insert(up_pt_idx);
            }
          }
        }
      }
      index_timer.Stop();
      LOG(INFO) << "points_with_indices size: " << points_with_indices.size();

      timing::Timer extract_timer("feature_extractor");  // around 0.01s

      // std::stod(fs_with_fn.filename) * 1e-9
      std::unique_ptr<Frame> keyframe_ptr =
          std::make_unique<Frame>(reader.gray_image, fs_with_fn.filename, feature_extractor.get(), &voc);
      Frame *pF = keyframe_ptr.get();
      pF->SetPose(Converter::toCvMat(curr_pose.inverse()));

      LOG(INFO) << "id: " << id << ", pF->mnId: " << pF->mnId << " N? " << pF->N;

      const std::vector<cv::KeyPoint> &keypoints = pF->mvKeysUn;
      const cv::Mat &features = pF->mDescriptors;
      extract_timer.Stop();

      desc_map.all_keyframes.insert(std::make_pair(keyframe_ptr->mnId, std::move(keyframe_ptr)));

      timing::Timer desc_map_timer("map_build");
      std::vector<std::vector<unsigned int>> neighbor_indices;

      desc_map.SetPointsAndFrame(points_with_indices, pF, neighbor_indices);

      {
        int cnt_mp = 0, cnt_obs = 0;
        for (auto &&id_kf: desc_map.all_keyframes) {
          auto &&kf = id_kf.second;
          for (auto &&pMP :kf->mvpMapPoints) {
            if (pMP) {
              ++cnt_mp;
            }
          }
          if (desc_map.mbFixMap) kf->mvpMapPoints.clear();
        }
        for (auto &&idx_map_points: desc_map.map_idx_map_points) {
          for (auto &&mp:idx_map_points.second) {
            cnt_obs += mp->GetObservations().size();
            if (desc_map.mbFixMap) mp->GetObservations().clear();
          }
        }
        LOG(INFO) << fmt::format(fmt::fg(fmt::color::deep_pink), "kf mp {}, obs {}", cnt_mp, cnt_obs);
      }

      desc_map_timer.Stop();

      // Let's validate the selected points!
      if (FLAGS_validate) {
        timing::Timer validate_timer("map_build.validate");
        std::set<unsigned int> already_added;
        neighbor_points.clear();
        for (int f_id = 0; f_id < neighbor_indices.size(); ++f_id) {
          auto &&v_n_idx = neighbor_indices[f_id];
          bool first_flag = true;
          for (auto &&n_idx : v_n_idx) {
            if (!already_added.count(n_idx)) {
              cv::Point3f worldPos(desc_map.GetGlobalPoint(n_idx));
              Eigen::Vector3f p_global(worldPos.x, worldPos.y, worldPos.z);
              neighbor_points.push_back(p_global);
              already_added.insert(n_idx);
            }
          }
        }

        associated_points.clear();
        for (auto &&map_point: desc_map.mvpMapPointMatches) {
          if (map_point) {
            cv::Mat world_pos = map_point->GetWorldPos();
            associated_points.emplace_back(world_pos.at<float>(0), world_pos.at<float>(1), world_pos.at<float>(2));
          }
        }
        LOG(INFO) << fmt::format(fmt::fg(fmt::color::coral),
                                 "desc_map.mvpMapPointMatches {}, associated_points {}, neighbor_points {}",
                                 desc_map.mvpMapPointMatches.size(), associated_points.size(), neighbor_points.size());

        PnPsolver solver(*pF, desc_map.mvpMapPointMatches);
        solver.SetRansacParameters(0.99, 10, 300, 4, 0.5, 5.991 * 3);
        std::vector<bool> vbInliers;
        int nInliers;
        bool bNoMore;
        cv::Mat Tcw = solver.iterate(200, bNoMore, vbInliers, nInliers);

        {
          auto &&CntMapPoints = [](std::unordered_map<unsigned int, std::vector<MapPoint * >> &m) {
            int cnt = 0;
            for (auto &&item:m)
              cnt += item.second.size();
            return cnt;
          };
          LOG(INFO) << fmt::format(fmt::fg(fmt::color::yellow),
                                   "mvpMapPointMatches size: {}, map points with desc {}, map_idx_map_points {}",
                                   desc_map.mvpMapPointMatches.size(),
                                   CntMapPoints(desc_map.map_idx_map_points),
                                   desc_map.map_idx_map_points.size());
        }

        if (!Tcw.empty()) {
          Eigen::Matrix4f pnp_pose_mat;
          cv::cv2eigen(Tcw.inv(), pnp_pose_mat);
          pnp_pose.setRotationMatrix(pnp_pose_mat.block<3, 3>(0, 0));
          pnp_pose.translation() = pnp_pose_mat.block<3, 1>(0, 3);
        }

        desc_points.clear();
        for (auto &&idx_map_points : desc_map.map_idx_map_points) {
          cv::Mat world_pos = idx_map_points.second.front()->GetWorldPos();
          Eigen::Vector3f p_global(world_pos.at<float>(0), world_pos.at<float>(1), world_pos.at<float>(2));
          desc_points.push_back(p_global);
        }

        old_neighbor_points.clear();
        for (auto &&map_point : pF->mvpMapPoints) {
          if (map_point) {
            for (auto &id: map_point->neighbor_points_global_ids) {
              cv::Point3f pn = desc_map.GetGlobalPoint(id);
              old_neighbor_points.emplace_back(pn.x, pn.y, pn.z);
            }
          }
        }

      }

      LOG(INFO) << timing::Timing::Print();

      timing::Timer vis_timer("vis");
      cv::Mat cv_plot;
      cv::cvtColor(reader.gray_image, cv_plot, cv::COLOR_GRAY2RGB);
      cv::Mat cv_black = cv::Mat::zeros(cv_plot.size(), cv_plot.type());

      cv::drawKeypoints(cv_plot, keypoints, cv_plot,
                        cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
      // cv::drawKeypoints(cv_black, keypoints, cv_black, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
      vis_timer.Stop();

      cv::namedWindow("cv_plot");
      cv::imshow("cv_plot", cv_plot);
      cv::waitKey(5);

      cv::Mat dist_img_f_vis = (dist_img_f / 10);
      textures[GPUTexture::RGB]->texture->Upload(cv_plot.data, GL_RGB, GL_UNSIGNED_BYTE);
      textures["Model"]->texture->Upload(dist_img_f_vis.data, GL_LUMINANCE, GL_FLOAT);

      if (++id >= all_frames_with_fn.size()) {
        LOG(INFO) << "finished";
      }

      trajectories.emplace_back(curr_pose.translation().cast<float>());
      relocalization_traj.emplace_back(pnp_pose.translation().cast<float>());

    }

    if (gui.draw_global_model->Get()) {
      global_model.RenderPointCloud(gui.s_cam.GetProjectionModelViewMatrix(),
                                    gui.draw_normals->Get(),
                                    gui.draw_colors->Get());
    }

    gui.DrawWorldPoints(neighbor_points, 1, Eigen::Vector3f(1, 0, 0));
    gui.DrawWorldPoints(associated_points, 10, Eigen::Vector3f(1, 1, 0));
    gui.DrawWorldPoints(desc_points, 4, Eigen::Vector3f(0, 1, 0));
    gui.DrawWorldPoints(old_neighbor_points, 3, Eigen::Vector3f(0, 1, 1));

    if (gui.draw_trajectory->Get()) {
      glLineWidth(3);
      glColor3f(1, 0, 0);
      pangolin::glDrawLineStrip(trajectories);
      glColor3f(1, 1, 0);
      pangolin::glDrawLineStrip(relocalization_traj);
      glColor3f(0, 0, 0);
      glLineWidth(1);
    }

    if (gui.followPose->Get()) {
      gui.FollowAbsPose(curr_pose.matrix().cast<float>());
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

        KeyFrameDatabase kf_db(&voc);
        kf_db.BuildInvertedFile(desc_map_tmp.all_keyframes);

        LOG(INFO) << (VerifyDescMap(desc_map, desc_map_tmp) ? "Verified" : "Not verified");
      }
    }

    if (pangolin::Pushed(*gui.run_opt)) {
      LOG(INFO) << "RUN OPT";
      desc_map.DoOptimization(FLAGS_run_opt);
    }

    glColor3f(0, 0, 1);
    gui.DrawFrustum(curr_pose.matrix().cast<float>());
    glColor3f(1, 1, 1);

    glColor3f(0, 1, 1);
    gui.DrawFrustum(pnp_pose.matrix());
    glColor3f(1, 1, 1);

    gui.DisplayImg("Model", textures["Model"], true);
    gui.DisplayImg(GPUTexture::RGB, textures[GPUTexture::RGB], true);

    gui.dataIdx->operator=(id);
    gui.PostCall();
  }

  delete vbo;

  return 0;
}