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

#include "map_render/map_render.h"
#include "relocalization/feature_extractor.h"
#include "relocalization/feature_matcher.h"
#include "relocalization/vocabulary_binary.h"
#include "relocalization/desc_map.h"
#include "relocalization/pnp_solver.h"
#include "relocalization/converter.h"
#include "relocalization/visualization/drawer.h"
#include "relocalization/relocalization_config.h"
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
DEFINE_string(load,
              "/mnt/HDD/Datasets/Visual/EuRoC/V1_02_medium/mav0/processed_data/left_pinhole/relocalization/desc_map.bin",
              "Map to load");

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_alsologtostderr = true;
  FLAGS_colorlogtostderr = true;

  std::vector<FrameShellWithFn> all_frames_with_fn = LoadTum(FLAGS_traj_path);

  EurocReader reader(FLAGS_path);
  RelocalizationConfig yaml_loader(reader.config_path);
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

  LOG(INFO) << "min_common_weight: " << yaml_loader.min_common_weight;

  ORBextractor feature_extractor(500, 1.2, 5, 20, 7);
  OrbVocabularyBinary voc;
  voc.loadFromBinaryFile(FLAGS_voc_path);

  GUI gui(FLAGS_offscreen, false);
  float point_ratio = 0;
  gui.dataIdx->Meta().range[1] = all_frames_with_fn.size();
  gui.pause->operator=(FLAGS_pause);

  pangolin::GlBuffer *vbo = map_loader.GetVbo();
  LOG(INFO) << vbo->SizeBytes() << " size bytes";

  GlobalModel global_model;
  global_model.Initialization(*vbo);

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

  if (boost::filesystem::exists(FLAGS_load)) {
    timing::Timer load_timer("load");
    desc_map.LoadMap(FLAGS_load);
    desc_map.SetFixMap(true);
  }

  CliqueMap clique_map(&voc);
  if (boost::filesystem::exists("/tmp/cliques.bin")) {
    timing::Timer load_timer("load");
    clique_map.Load("/tmp/cliques.bin");
  } else {
    clique_map.ComputeCliques(desc_map.all_map_points, yaml_loader.min_common_weight);
    clique_map.Save("/tmp/cliques.bin");
  }

  std::vector<Eigen::Vector3f> rand_colors;
  for (auto &&c:clique_map.mCliques) {
    rand_colors.push_back(Eigen::Vector3f::Random());
  }

  typedef typename std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > vVec3f;
  vVec3f trajectories, relocalization_traj;
  vVec3f neighbor_points, old_neighbor_points;
  vVec3f associated_points;
  vVec3f desc_points;
  vVec3f connected_points, rand_points;

  double num_frames = 0, num_found = 0;

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
      cv::Mat dist_img_f(hG[0], wG[0], CV_32FC1);
      pangolin::TypedImage pgl_dist_img;

      cv::Mat cv_plot;
      cv::cvtColor(reader.gray_image, cv_plot, cv::COLOR_GRAY2RGB);

      timing::Timer extract_timer("feature_extractor");  // around 0.01s

      std::unique_ptr<Frame> keyframe_ptr =
          std::make_unique<Frame>(reader.gray_image, fs_with_fn.filename, &feature_extractor, &voc);
      Frame *pF = keyframe_ptr.get();
      pF->SetPose(Converter::toCvMat(curr_pose.inverse()));

      const std::vector<cv::KeyPoint> &keypoints = pF->mvKeysUn;
      const cv::Mat &features = pF->mDescriptors;
      extract_timer.Stop();

      desc_map.all_keyframes.insert(std::make_pair(keyframe_ptr->mnId, std::move(keyframe_ptr)));

      timing::Timer desc_map_timer("desc_map");
      std::vector<std::vector<unsigned int>> neighbor_indices;

      std::vector<PointWithIndex> points_with_indices;

      if (yaml_loader.gt_projection) {
        timing::Timer index_timer("index");
        index_map.SynthesizeDepth(curr_pose.matrix().cast<float>(), global_model.Model(), 100);
        index_map.DepthTex()->texture->Download(pgl_dist_img);
        dist_img_f = cv::Mat(hG[0], wG[0], CV_32FC1, pgl_dist_img.ptr);

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
        std::set<unsigned int> set_idx_already_added;
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
          }  // for x
        }  // for y
        index_timer.Stop();
        // Using distance and cliques
        {
          std::set<Clique *> sAlreadyAdded;
          for (auto &&c:clique_map.mCliques) {
            if (pF->IsInFrustum(c->mMeanWorldPos)) {
              sAlreadyAdded.insert(c.get());
            }
          }
          // cv::Mat mPoint = (cv::Mat_<float>(3, 1) << 0, 0, 0);
          // for (auto &&pwi:points_with_indices) {
          //   cv::Mat point(desc_map.GetGlobalPoint(pwi.index_in_surfel_map));
          //   mPoint += point;
          // }
          // mPoint /= points_with_indices.size();
          // for (auto &&c:clique_map.mCliques)
          //   if (cv::norm(mPoint - c->mMeanWorldPos) < 1) {
          //     sAlreadyAdded.insert(c.get());
          //   }

          points_with_indices.clear();
          std::vector<MapPoint *>
              priority_vpMapPointMatches = std::vector<MapPoint *>(pF->N, static_cast<MapPoint *>(NULL));
          for (auto &&clique : sAlreadyAdded) {
            auto &&mIds_in_clique = clique->mvNId;
            for (auto &&mId :mIds_in_clique) {
              points_with_indices.emplace_back(0, 0, desc_map.all_map_points[mId]->idx_in_surfel_map);
            }
          }
          desc_map.ComputeLocalBoW(points_with_indices, pF);
        }

        // Using all points
        // desc_map.ComputeLocalBoW(points_with_indices, pF);
      } else {
        // TODO: move it to a detecion function
        pF->ComputeBoW();
        std::vector<float> scores;
        for (auto &&pClique:clique_map.mCliques) {
          scores.push_back(voc.score(pClique->mBowVec, pF->mBowVec));
        }
        std::vector<Clique *> vpCandidates = clique_map.DetectRelocalizationCandidates(pF);
        LOG(INFO) << "vpRelocCandidates: " << vpCandidates.size();

        // TODO: Search from best clique first? then others?
        // TODO: Search by inverted map for the cliques?
        points_with_indices.clear();
        std::vector<MapPoint *>
            priority_vpMapPointMatches = std::vector<MapPoint *>(pF->N, static_cast<MapPoint *>(NULL));
        for (auto &&clique = vpCandidates.rbegin(); clique != vpCandidates.rend(); ++clique) {
          std::vector<PointWithIndex> tmp_pwi;
          auto &&mIds_in_clique = (*clique)->mvNId;
          for (auto &&mId :mIds_in_clique) {
            points_with_indices.emplace_back(0, 0, desc_map.all_map_points[mId]->idx_in_surfel_map);
            tmp_pwi.emplace_back(0, 0, desc_map.all_map_points[mId]->idx_in_surfel_map);
          }
          desc_map.ComputeLocalBoW(tmp_pwi, pF);
          for (int f_id = 0; f_id < pF->N; ++f_id) {
            if (!priority_vpMapPointMatches[f_id] && desc_map.mvpMapPointMatches[f_id])
              priority_vpMapPointMatches[f_id] = desc_map.mvpMapPointMatches[f_id];
          }
        }
        desc_map.mvpMapPointMatches = priority_vpMapPointMatches;
      }

      // desc_map.SetPointsAndFrame(points_with_indices, pF, neighbor_indices);
      // Replace the SetPointsAndFrame to obtain desc_map.mvpMapPointMatches

      // Visualize matches
      {
        cv::Mat out_img;
        DrawMapPointMatches(pF, desc_map.mvpMapPointMatches, cv_plot, out_img);
        cv::namedWindow("matches");
        cv::imshow("matches", out_img);
        cv::waitKey(5);
      }

      LOG(INFO) << "points_with_indices size: " << points_with_indices.size();

      neighbor_points.clear();
      connected_points.clear();
      rand_points.clear();
      {
        auto &&vpMPM = desc_map.mvpLocalMapPoints; // desc_map.all_map_points
        int rand_int = RandomInt(0, vpMPM.size() - 1);
        MapPoint *pMP;
        auto &&mpit = vpMPM.begin();
        for (int mpid = 0; mpid < rand_int; ++mpid) pMP = *(mpit++); // (mpit++)->second.get()
        cv::Point3f rp = desc_map.GetGlobalPoint(pMP->idx_in_surfel_map);
        rand_points.emplace_back(rp.x, rp.y, rp.z);
        for (auto &&nid:pMP->neighbor_points_global_ids) {
          cv::Point3f np = desc_map.GetGlobalPoint(nid);
          neighbor_points.emplace_back(np.x, np.y, np.z);
        }
      }

      {
        int cnt_mp = 0, cnt_obs = 0;
        for (auto &&id_kf: desc_map.all_keyframes) {
          auto&& kf = id_kf.second;
          for (auto &&pMP :kf->mvpMapPoints) {
            if (pMP) {
              ++cnt_mp;
            }
          }
          if (desc_map.mbFixMap) kf->mvpMapPoints.clear();  // we do not use this data
        }
        for (auto &&idx_map_points: desc_map.map_idx_map_points) {
          for (auto &&mp:idx_map_points.second) {
            cnt_obs += mp->GetObservations().size();
            if (desc_map.mbFixMap) mp->GetObservations().clear(); // we do not use this data
          }
        }
        LOG(INFO) << fmt::format(fmt::fg(fmt::color::deep_pink), "kf mp {}, obs {}", cnt_mp, cnt_obs);
      }

      desc_map_timer.Stop();

      // Let's validate the selected points!
      {
        PnPsolver solver(*pF, desc_map.mvpMapPointMatches);
        solver.SetRansacParameters(0.99, 10, 300, 4, 0.5, 5.991);
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
          auto &&ValidPoints = [](std::vector<MapPoint *> &m) {
            int cnt = 0;
            for (auto &&item:m)
              if (item) cnt += 1;
            return cnt;
          };
          LOG(INFO) << fmt::format(fmt::fg(fmt::color::yellow),
                                   "mvpMapPointMatches valid: {}, map points with desc {}, map_idx_map_points {}",
                                   ValidPoints(desc_map.mvpMapPointMatches),
                                   CntMapPoints(desc_map.map_idx_map_points),
                                   desc_map.map_idx_map_points.size());
        }

        if (!Tcw.empty()) {
          Eigen::Matrix4f pnp_pose_mat;
          cv::cv2eigen(Tcw.inv(), pnp_pose_mat);
          pnp_pose.setRotationMatrix(pnp_pose_mat.block<3, 3>(0, 0));
          pnp_pose.translation() = pnp_pose_mat.block<3, 1>(0, 3);
          relocalization_traj.emplace_back(pnp_pose.translation().cast<float>());
          num_found += 1;
        }
        num_frames += 1;
        LOG(INFO) << fmt::format("pose recal: {:.1f}%", num_found / num_frames * 100);
      }

      desc_points.clear();
      // for (auto &&pwi:points_with_indices) {
      //   cv::Point3f world_pos = desc_map.GetGlobalPoint(pwi.index_in_surfel_map);
      //   desc_points.emplace_back(world_pos.x, world_pos.y, world_pos.z);
      // }
      for (auto &&pMP:desc_map.mvpMapPointMatches) {
        if (!pMP) continue;
        cv::Mat world_pos = pMP->GetWorldPos();
        Eigen::Vector3f p_global(world_pos.at<float>(0), world_pos.at<float>(1), world_pos.at<float>(2));
        desc_points.push_back(p_global);
      }
      // for (auto &&idx_map_points : desc_map.map_idx_map_points) {
      //   cv::Mat world_pos = idx_map_points.second.front()->GetWorldPos();
      //   Eigen::Vector3f p_global(world_pos.at<float>(0), world_pos.at<float>(1), world_pos.at<float>(2));
      //   desc_points.push_back(p_global);
      // }

      LOG(INFO) << timing::Timing::Print();

      cv::Mat cv_black = cv::Mat::zeros(cv_plot.size(), cv_plot.type());

      cv::drawKeypoints(cv_plot, keypoints, cv_plot,
                        cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
      // cv::drawKeypoints(cv_black, keypoints, cv_black, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

      cv::Mat dist_img_f_vis = (dist_img_f / 10);
      textures[GPUTexture::RGB]->texture->Upload(cv_plot.data, GL_RGB, GL_UNSIGNED_BYTE);
      if (yaml_loader.gt_projection) {
        textures["Model"]->texture->Upload(dist_img_f_vis.data, GL_LUMINANCE, GL_FLOAT);
      }

      if (++id >= all_frames_with_fn.size()) {
        LOG(INFO) << "finished";
      }

      trajectories.emplace_back(curr_pose.translation().cast<float>());

    }

    if (gui.draw_global_model->Get()) {
      global_model.RenderPointCloud(gui.s_cam.GetProjectionModelViewMatrix(),
                                    gui.draw_normals->Get(),
                                    gui.draw_colors->Get());
    }

    gui.DrawWorldPoints(neighbor_points, 1, Eigen::Vector3f(1, 0, 0));
    gui.DrawWorldPoints(rand_points, 10, Eigen::Vector3f(1, 1, 0));
    gui.DrawWorldPoints(desc_points, 3, Eigen::Vector3f(0, 1, 0));
    gui.DrawWorldPoints(connected_points, 7, Eigen::Vector3f(0, 1, 1));

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