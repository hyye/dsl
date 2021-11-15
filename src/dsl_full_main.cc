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

#include "full_system/full_system.h"
#include "map_render/map_render.h"
#include "tool/euroc_reader.h"

#include "relocalization/sp_extractor.h"

#include "map_render/map_render.h"
#include "relocalization/feature_extractor.h"
#include "relocalization/vocabulary_binary.h"
#include "relocalization/struct/key_frame_database.h"
#include "relocalization/struct/vlad_database.h"
#include "relocalization/desc_map.h"
#include "relocalization/pnp_solver.h"
#include "relocalization/converter.h"
#include "relocalization/visualization/drawer.h"
#include "relocalization/relocalization_config.h"

using namespace dsl;
using namespace dsl::relocalization;

DEFINE_string(path, "", "input path");
DEFINE_string(config, "config_full.yaml", "config filename");
DEFINE_bool(baseline, false, "enable baseline");
DEFINE_bool(rgb, false, "load rgb");
// DEFINE_bool(set_prior, false, "set first pose prior");
DEFINE_bool(offscreen, false, "offscreen");
DEFINE_bool(no_marg, false, "no_marg");
DEFINE_bool(scaling, false, "scaling");
DEFINE_bool(pause, false, "pause");
DEFINE_bool(auto_exit, false, "auto_exit");

std::string ToTum(std::string time, const SE3 &se3) {
  std::stringstream ss;
  Eigen::Quaterniond q = se3.unit_quaternion();
  Vec3 p = se3.translation();
  ss << time.substr(0, 10) << "." << time.substr(10, 9) << " " << p.x() << " "
     << p.y() << " " << p.z() << " " << q.x() << " " << q.y() << " " << q.z()
     << " " << q.w() << std::endl;
  return ss.str();
}

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_alsologtostderr = true;

  // settingDesiredImmatureDensity = 750;
  // settingDesiredPointDensity = 1000;

  EurocReader reader(FLAGS_path, FLAGS_config);
  RelocalizationConfig yaml_loader(reader.config_path);
  MapLoader map_loader(yaml_loader.pcd_file, 1,
                       yaml_loader.intensity_scalar, yaml_loader.rgb_scalar, yaml_loader.surfel_size,
                       FLAGS_rgb);
  Mat33f K;
  K << yaml_loader.gamma1, 0, yaml_loader.u0, 0, yaml_loader.gamma2,
      yaml_loader.v0, 0, 0, 1;
  SetGlobalCalib(yaml_loader.image_width, yaml_loader.image_height, K,
                 yaml_loader.xi);
  boost::filesystem::path mask_path = boost::filesystem::path(FLAGS_path) / "mask.png";
  if (boost::filesystem::exists(mask_path)) {
    SetGlobalMask(mask_path.string());
    LOG(WARNING) << "MASK LOADED: " << mask_path;
  }

  settingBaseline = FLAGS_baseline;
  settigEnableMarginalization = !FLAGS_no_marg;
  settigNoScalingAtOpt = !FLAGS_scaling;

  settingReTrackThreshold = yaml_loader.retrack_threshold;

  use_superpoint = yaml_loader.use_superpoint;

  if (settingBaseline) {
    LOG(WARNING) << "baseline";
  }

  std::unique_ptr<ORBextractor> feature_extractor;
  OrbVocabularyBinary voc;
  DescMap desc_map;
  desc_map.config = yaml_loader.desc_map_config;
  desc_map.SetGlobalVertices(map_loader.vertices, map_loader.vertex_size, map_loader.ptc_size);

  if (boost::filesystem::exists(yaml_loader.database_map_path)) {
    timing::Timer load_timer("load");
    desc_map.LoadMap(yaml_loader.database_map_path);
    desc_map.SetFixMap(true);
    desc_map.keyframe_db.SetORBvocabulary(&voc);
    desc_map.keyframe_db.BuildInvertedFile(desc_map.all_keyframes);
    if (desc_map.config.enhanced_points || desc_map.config.use_all_points_in_active_search) {
      desc_map.EnhanceKeyframePoints();
      LOG(INFO) << desc_map.map_idx_enhanced_map_points.size();
    }
    desc_map.SetVLADPath(yaml_loader.database_vlad_path, yaml_loader.query_vlad_path);

    if (!use_superpoint) {
      std::vector<Frame *> vpKFs;
      for (auto &&it: desc_map.all_keyframes) {
        vpKFs.push_back(it.second.get());
      }
      desc_map.ComputeFrameNodeIds(vpKFs, desc_map.config.levelsup);
      desc_map.BuildPrioritizedNodes(desc_map.config.levelsup);
    }
  }

  if (use_superpoint) {
    SPExtractor::PyTorchDevice pytorch_device =
        yaml_loader.pytorch_device == "CUDA" ? SPExtractor::PyTorchDevice::CUDA : SPExtractor::PyTorchDevice::CPU;
    feature_extractor = std::make_unique<SPExtractor>(1000,
                                                      yaml_loader.superpoint_path,
                                                      pytorch_device,
                                                      yaml_loader.superpoint_conf_thresh);
  } else {
    feature_extractor = std::make_unique<ORBextractor>(yaml_loader.max_num_features, 1.2, 8, 20, 7);
    voc.loadFromBinaryFile(yaml_loader.database_voc_path);
    desc_map.SetVocabularyBinary(&voc);
  }
  LOG(INFO) << (use_superpoint ? "use sp" : "no sp");

  GUI gui(FLAGS_offscreen, false);
  std::vector<std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>>
      trajectories;
  trajectories.resize(1);
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>
      colors;
  colors.emplace_back(Eigen::Vector3f::Random());
  float point_ratio = 0;
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>
      photo_points;
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>
      homo_points;
  gui.dataIdx->Meta().range[1] = reader.filenames.size();
  gui.pause->operator=(FLAGS_pause);
  gui.draw_colors->operator=(false);
  gui.draw_trajectory->operator=(true);

  pangolin::GlBuffer *vbo = map_loader.GetVbo();
  LOG(INFO) << vbo->SizeBytes() << " size bytes";

  GlobalModel global_model;
  global_model.Initialization(*vbo);

  PangolinUtils new_vn;
  IndexMap index_map;
  Eigen::Affine3f kf_transform, ex_trans;
  // ex_trans = yaml_loader.T_lc;
  // kf_transform = yaml_loader.initial_pose * ex_trans;

  int id = yaml_loader.fast_forward;
  reader.ReadImage(id);
  cv::Mat cv_imgf;
  reader.gray_image.convertTo(cv_imgf, CV_32FC1);
  ImageAndExposure img(cv_imgf.cols, cv_imgf.rows, 1);
  memcpy(img.image.data(), cv_imgf.data,
         sizeof(float) * cv_imgf.cols * cv_imgf.rows);

  std::unique_ptr<FullSystem> full_system = std::make_unique<FullSystem>();
  // full_system->SetNoPosePrior(!FLAGS_set_prior);
  std::ofstream odom_file, kf_file;

  auto t = std::time(nullptr);
  auto tm = *std::localtime(&t);

  std::ostringstream oss;
  oss << std::put_time(&tm, "%y%m%d%H%M%S");
  std::string time_str = oss.str();
  boost::filesystem::path output_folder("/tmp/" + yaml_loader.output_folder);
  if (boost::filesystem::create_directories(output_folder)) {
    LOG(INFO) << "Directory Created: " << output_folder;
  }
  odom_file.open((output_folder / ("all_odom_" + time_str + ".tum")).string());
  kf_file.open((output_folder / ("kf_odom_" + time_str + ".tum")).string());

  std::map<std::string, GPUTexture *> textures;
  textures[GPUTexture::RGB] = new GPUTexture(wG[0], hG[0], GL_RGBA, GL_RGB,
                                             GL_UNSIGNED_BYTE, true, true);
  SE3f curr_pose;
  std::vector<SE3f> kf_poses;
  cv::Mat cv_plot, init_plot;
  std::unique_ptr<std::thread> plot_thread;
  std::mutex plot_mutex;
  std::condition_variable cond_var;
  bool plot_terminate = false;
  bool view_set = false;

  // TODO: move to desc_map
  bool last_relocOK = false, first_reloc = false, system_init = false;
  SE3f last_pnp_pose, pnp_pose;

  while (!pangolin::ShouldQuit()) {
    gui.PreCall();

    if ((!gui.pause->Get() || pangolin::Pushed(*gui.step)) &&
        id < reader.filenames.size()) {

      reader.ReadImage(id);
      reader.gray_image.convertTo(cv_imgf, CV_32FC1);

      if (!full_system->initialized) {
        std::unique_ptr<Frame> keyframe_ptr =
            std::make_unique<Frame>(reader.gray_image, reader.filenames[id], feature_extractor.get(), &voc);
        Frame *pF = keyframe_ptr.get();

        bool relocOK = desc_map.Relocalization(pF);
        if (!relocOK) {
          pF->mTcw_pnp = cv::Mat();
        }

        if (!pF->mTcw_pnp.empty()) {
          Eigen::Matrix4f pnp_pose_mat, raw_pnp_pose_mat;
          cv::cv2eigen(pF->mTcw_pnp.inv(), pnp_pose_mat);
          pnp_pose.setRotationMatrix(pnp_pose_mat.block<3, 3>(0, 0));
          pnp_pose.translation() = pnp_pose_mat.block<3, 1>(0, 3);

          if (first_reloc && (last_pnp_pose.translation() - pnp_pose.translation()).norm()
              < desc_map.config.translation_threshold) {
            system_init = true;
            kf_transform.linear() = pnp_pose.rotationMatrix().cast<float>();
            kf_transform.translation() = pnp_pose.translation().cast<float>();
            LOG(WARNING) << "INITIALIZED!!!";
            // gui.pause->operator=(true);
            // NOTE: Visualization
            {
              EurocReader db_reader(yaml_loader.database_dataset_path);
              const std::vector<std::string> &db_fns = db_reader.filenames;
              if (desc_map.mnMatchKeyFrameDBId != 0) {
                db_reader.ReadImage(desc_map.all_keyframes[desc_map.mnMatchKeyFrameDBId]->mTimeStamp);

                std::vector<cv::KeyPoint> kpts1, kpts2, kpts_proj;
                std::vector<cv::DMatch> matches1to2;
                Frame *pKF = desc_map.all_keyframes[desc_map.mnMatchKeyFrameDBId].get();

                ConvertMatches(pF, pKF, desc_map.mvpMapPointMatches, kpts1, kpts2, matches1to2);

                ConvertMatchesToKpts(pF, desc_map.mvpMapPointMatches, pF->mTcw_pnp, kpts_proj);

                cv::drawMatches(reader.gray_image, kpts1, db_reader.gray_image, kpts2, matches1to2,
                                init_plot);
                for (auto &&kpt_it = kpts1.begin() + matches1to2.size(); kpt_it != kpts1.end(); ++kpt_it) {
                  cv::drawMarker(init_plot, kpt_it->pt, cv::Scalar(255, 255, 255), cv::MARKER_CROSS);
                }
                for (auto &&kpt : kpts_proj) {
                  cv::drawMarker(init_plot, kpt.pt, cv::Scalar(255, 0, 0), cv::MARKER_SQUARE, 10, 1);
                }

                std::string
                    draw_txt = "left: current frame, right: DB KF " + std::to_string(desc_map.mnMatchKeyFrameDBId);

                AddTextToImage(draw_txt, init_plot, 255, 255, 255);

                plot_thread = std::make_unique<std::thread>([&] {
                  cv::namedWindow("pair");
                  cv::imshow("pair", init_plot);
                  std::unique_lock<std::mutex> lk(plot_mutex);
                  cond_var.wait(lk, [&] {
                    cv::waitKey(10);
                    return plot_terminate;
                  });
                });

              }
            }
          }
          if (!first_reloc) first_reloc = true;
          last_relocOK = true;
          last_pnp_pose = pnp_pose;
        } else {
          last_relocOK = false;
        }

        if (!system_init) {
          ++id;
          continue;
        }

        // NOTE: initialized
        std::vector<float> all_dist(wG[0] * hG[0]);
        pangolin::TypedImage pgl_dist_img;

        index_map.SynthesizeDepth(kf_transform.matrix(), global_model.Model(), 100);
        index_map.PredictGlobalIndices(kf_transform.matrix(), global_model.Model(),
                                       100);
        new_vn.SetVertexNormal(index_map.VertGlobalTex()->texture,
                               index_map.NormalGlobalTex()->texture);

        index_map.DepthTex()->texture->Download(pgl_dist_img);

        cv::Mat depth_img_f = cv::Mat(hG[0], wG[0], CV_32FC1, pgl_dist_img.ptr);
        all_dist.assign(depth_img_f.begin<float>(), depth_img_f.end<float>());

        SE3 initial_cam;
        initial_cam.setQuaternion(
            Eigen::Quaterniond(kf_transform.cast<double>().linear()));
        initial_cam.translation() = kf_transform.translation().cast<double>();

        full_system->AddActiveFrame(img, id, all_dist, initial_cam,
                                    (Eigen::Vector4f *) new_vn.vertex_img.ptr,
                                    (Eigen::Vector4f *) new_vn.normal_img.ptr);

        odom_file << ToTum(reader.filenames[id],
                           full_system->all_frame_shells.back()->cam_to_world);
        kf_file << ToTum(reader.filenames[id],
                         full_system->all_frame_shells.back()->cam_to_world);
        if (!view_set) {
          gui.FollowAbsPose(kf_transform.matrix());
          view_set = true;
        }
      }

      // FIXME: replace id to HasMore function
      if (++id >= reader.filenames.size()) {
        LOG(INFO) << "finished";
        if (FLAGS_auto_exit) break;
        continue;
      }
      memcpy(img.image.data(), cv_imgf.data,
             sizeof(float) * cv_imgf.cols * cv_imgf.rows);

      LOG(INFO) << "id: " << id;
      full_system->AddActiveFrame(img, id, std::vector<float>(), SE3(),
                                  (Eigen::Vector4f *) new_vn.vertex_img.ptr,
                                  (Eigen::Vector4f *) new_vn.normal_img.ptr);

      if (full_system->is_keyframe) {
        kf_transform.matrix() = full_system->frame_hessians.back()
            ->PRE_cam_to_world.matrix()
            .cast<float>();
        index_map.PredictGlobalIndices(kf_transform.matrix(),
                                       global_model.Model(), 100);
        new_vn.SetVertexNormal(index_map.VertGlobalTex()->texture,
                               index_map.NormalGlobalTex()->texture);

        kf_file << ToTum(reader.filenames[id],
                         full_system->all_frame_shells.back()->cam_to_world);
      }

      curr_pose =
          full_system->all_frame_shells.back()->cam_to_world.cast<float>();
      kf_poses.clear();

      for (auto &&fh : full_system->frame_hessians) {
        kf_poses.emplace_back(fh->shell->cam_to_world.cast<float>());
      }

      {
        photo_points.clear();
        homo_points.clear();
        std::vector<cv::Point3f> tracked_points_blue;
        std::vector<cv::Point3f> tracked_points_green;

        for (auto &&fh : full_system->frame_hessians) {
          SE3f fh_pose = fh->PRE_cam_to_world.cast<float>();
          for (auto &&ph : fh->point_hessians) {
            float host_u = ph->u;
            float host_v = ph->v;
            float idist = ph->idist;
            Eigen::Vector3f p_cam = LiftToSphere(
                host_u, host_v, fxiG[0], fyiG[0], cxiG[0], cyiG[0], xiG);
            p_cam /= idist;
            Eigen::Vector3f p_w = fh_pose * p_cam;
            Eigen::Vector3f p_curr = SpaceToPlane(
                curr_pose.inverse() * p_w, fxG[0], fyG[0], cxG[0], cyG[0], xiG);
            if (ph->convereged_ph_idist) {
              homo_points.emplace_back(p_w);
              tracked_points_green.emplace_back(p_curr.x(), p_curr.y(), idist);
            } else {
              photo_points.emplace_back(p_w);
              tracked_points_blue.emplace_back(p_curr.x(), p_curr.y(), idist);
            }
          }
        }
        cv::cvtColor(reader.gray_image, cv_plot, cv::COLOR_GRAY2RGB);
        for (auto &&cv_p : tracked_points_blue) {
          cv::circle(cv_plot, cv::Point2f(cv_p.x, cv_p.y), 1,
                     cv::Scalar(0, 0, 255), 2);
        }
        for (auto &&cv_p : tracked_points_green) {
          cv::circle(cv_plot, cv::Point2f(cv_p.x, cv_p.y), 1,
                     cv::Scalar(0, 255, 0), 2);
        }

        AddTextToImage("rmse: " + std::to_string(full_system->last_rmse), cv_plot, 255, 255, 255);
        textures[GPUTexture::RGB]->texture->Upload(cv_plot.data, GL_RGB,
                                                   GL_UNSIGNED_BYTE);
      }

      odom_file << ToTum(reader.filenames[id],
                         full_system->all_frame_shells.back()->cam_to_world);

      int num_points = 0, num_homo = 0, num_valid = 0;
      for (auto &&fh : full_system->frame_hessians) {
        for (auto &&ph : fh->point_hessians) {
          ++num_points;
          if (ph->convereged_ph_idist) {
            ++num_homo;
          }
          if (ph->valid_plane) {
            ++num_valid;
          }
        }
      }
      point_ratio = 1.0 * num_homo / num_points;
      // DLOG(INFO) << "@@@@@@@ ratio: " << point_ratio << "; " << num_valid << ":"
      //            << num_points;

      trajectories.back().emplace_back(full_system->all_frame_shells.back()
                                    ->cam_to_world.translation()
                                    .cast<float>());
    }

    if (pangolin::Pushed(*gui.run_opt) || full_system->last_rmse > 10 || full_system->is_lost) {
      LOG(INFO) << "RESET!!!";
      // trajectories.clear();
      trajectories.resize(trajectories.size() + 1);
      colors.emplace_back(Eigen::Vector3f::Random());
      kf_poses.clear();
      full_system = std::make_unique<FullSystem>();
      last_relocOK = false, first_reloc = false, system_init = false;

      {
        std::lock_guard<std::mutex> lk(plot_mutex);
        plot_terminate = true;
      }
      cond_var.notify_all();
      if (plot_thread) plot_thread->join();
      {
        std::lock_guard<std::mutex> lk(plot_mutex);
        plot_terminate = false;
      }

      // LOG(INFO) << "SAVE FIGURES";
      // cv::Mat cv_out;
      // cv::cvtColor(cv_plot, cv_out, CV_RGB2BGR);
      // cv::imwrite("/tmp/tracked_points_" + time_str + ".jpg", cv_out);
    }
    cond_var.notify_all();

    if (gui.draw_global_model->Get()) {
      global_model.RenderPointCloud(gui.s_cam.GetProjectionModelViewMatrix(),
                                    gui.draw_normals->Get(),
                                    gui.draw_colors->Get());
    }

    if (gui.draw_trajectory->Get()) {
      glLineWidth(3);
      for (int id_traj = 0; id_traj < trajectories.size(); ++id_traj) {
        glColor3f(colors[id_traj].x(), colors[id_traj].y(), colors[id_traj].z());
        pangolin::glDrawLineStrip(trajectories[id_traj]);
        glColor3f(0, 0, 0);
      }
      // glColor3f(1, 0.647, 0);
      // glLineWidth(1);
      // pangolin::glDrawLineStrip(gt_trajectories);
      // glColor3f(0, 0, 0);
    }

    if (gui.draw_predict->Get()) {
      if (homo_points.size() > 0) {
        gui.DrawWorldPoints(homo_points, 7.0,
                            Eigen::Vector3f(0, 1, 0));  // green
      }

      if (photo_points.size() > 0) {
        gui.DrawWorldPoints(photo_points, 7.0, Eigen::Vector3f(0, 0, 1));
      }
    }

    if (gui.followPose->Get()) {
      gui.FollowAbsPose(curr_pose.matrix());
    }

    glColor3f(0, 0, 1);
    gui.DrawFrustum(curr_pose.matrix());
    glColor3f(1, 1, 1);
    glColor3f(1, 1, 0);
    for (auto &&kf_pose : kf_poses) {
      gui.DrawFrustum(kf_pose.matrix());
    }
    glColor3f(1, 1, 1);

    gui.DisplayImg(GPUTexture::RGB, textures[GPUTexture::RGB], true);

    gui.dataIdx->operator=(id);
    gui.percentageIdx->operator=(point_ratio);

    gui.PostCall();
  }

  {
    std::lock_guard<std::mutex> lk(plot_mutex);
    plot_terminate = true;
  }
  cond_var.notify_all();
  if (plot_thread) plot_thread->join();

  delete vbo;

  return 0;
}