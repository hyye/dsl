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

using namespace dsl;

DEFINE_string(path, "", "input path");
DEFINE_bool(baseline, false, "enable baseline");
DEFINE_bool(set_prior, false, "set first pose prior");
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

  EurocReader reader(FLAGS_path);
  YamlLoader yaml_loader(reader.config_path);
  MapLoader map_loader(yaml_loader.pcd_file, 1,
      yaml_loader.intensity_scalar, yaml_loader.rgb_scalar, yaml_loader.surfel_size);
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
  settigEnableMarginalization= !FLAGS_no_marg;
  settigNoScalingAtOpt = !FLAGS_scaling;
  settingReTrackThreshold = yaml_loader.retrack_threshold;

  if (settingBaseline) {
    LOG(WARNING) << "baseline";
  }

  GUI gui(FLAGS_offscreen, false);
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>
      trajectories;
  float point_ratio = 0;
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>
      photo_points;
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>
      homo_points;
  gui.dataIdx->Meta().range[1] = reader.filenames.size();
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

  int id = yaml_loader.fast_forward;
  reader.ReadImage(id);
  cv::Mat cv_imgf;
  reader.gray_image.convertTo(cv_imgf, CV_32FC1);
  ImageAndExposure img(cv_imgf.cols, cv_imgf.rows, 1);
  memcpy(img.image.data(), cv_imgf.data,
         sizeof(float) * cv_imgf.cols * cv_imgf.rows);

  FullSystem full_system;
  full_system.SetNoPosePrior(!FLAGS_set_prior);
  std::ofstream odom_file, kf_file;

  auto t = std::time(nullptr);
  auto tm = *std::localtime(&t);

  std::ostringstream oss;
  oss << std::put_time(&tm, "%y%m%d%H%M%S");
  std::string time_str = oss.str();
  boost::filesystem::path output_folder("/tmp/" + yaml_loader.output_folder);
  if(boost::filesystem::create_directories(output_folder)) {
    LOG(INFO) << "Directory Created: " << output_folder;
  }
  odom_file.open((output_folder / ("all_odom_" + time_str + ".tum")).string());
  kf_file.open((output_folder / ("kf_odom_" + time_str + ".tum")).string());

  std::vector<float> all_dist(wG[0] * hG[0]);
  pangolin::TypedImage pgl_dist_img;
  if (!settingBaseline || !full_system.initialized) {
    index_map.SynthesizeDepth(kf_transform.matrix(), global_model.Model(), 100);
    index_map.PredictGlobalIndices(kf_transform.matrix(), global_model.Model(),
                                   100);
    new_vn.SetVertexNormal(index_map.VertGlobalTex()->texture,
                           index_map.NormalGlobalTex()->texture);

    index_map.DepthTex()->texture->Download(pgl_dist_img);

    cv::Mat depth_img_f = cv::Mat(hG[0], wG[0], CV_32FC1, pgl_dist_img.ptr);
    all_dist.assign(depth_img_f.begin<float>(), depth_img_f.end<float>());
  }

  SE3 initial_cam;
  initial_cam.setQuaternion(
      Eigen::Quaterniond(kf_transform.cast<double>().linear()));
  initial_cam.translation() = kf_transform.translation().cast<double>();
  full_system.AddActiveFrame(img, id, all_dist, initial_cam,
                             (Eigen::Vector4f *)new_vn.vertex_img.ptr,
                             (Eigen::Vector4f *)new_vn.normal_img.ptr);

  odom_file << ToTum(reader.filenames[id],
                     full_system.all_frame_shells.back()->cam_to_world);
  kf_file << ToTum(reader.filenames[id],
                   full_system.all_frame_shells.back()->cam_to_world);

  gui.FollowAbsPose(kf_transform.matrix());
  std::map<std::string, GPUTexture *> textures;
  textures[GPUTexture::RGB] = new GPUTexture(wG[0], hG[0], GL_RGBA, GL_RGB,
                                             GL_UNSIGNED_BYTE, true, true);
  SE3f curr_pose;
  std::vector<SE3f> kf_poses;
  cv::Mat cv_plot;

  while (!pangolin::ShouldQuit()) {
    gui.PreCall();

    if ((!gui.pause->Get() || pangolin::Pushed(*gui.step)) &&
        id < reader.filenames.size()) {
      // FIXME: replace id to HasMore function
      if (++id >= reader.filenames.size()) {
        LOG(INFO) << "finished";
        if (FLAGS_auto_exit) break;
        continue;
      }

      reader.ReadImage(id);
      reader.gray_image.convertTo(cv_imgf, CV_32FC1);

      memcpy(img.image.data(), cv_imgf.data,
             sizeof(float) * cv_imgf.cols * cv_imgf.rows);

      LOG(INFO) << "id: " << id;
      full_system.AddActiveFrame(img, id, std::vector<float>(), SE3(),
                                 (Eigen::Vector4f *)new_vn.vertex_img.ptr,
                                 (Eigen::Vector4f *)new_vn.normal_img.ptr);

      if (full_system.is_keyframe) {
        kf_transform.matrix() = full_system.frame_hessians.back()
                                    ->PRE_cam_to_world.matrix()
                                    .cast<float>();
        index_map.PredictGlobalIndices(kf_transform.matrix(),
                                       global_model.Model(), 100);
        new_vn.SetVertexNormal(index_map.VertGlobalTex()->texture,
                               index_map.NormalGlobalTex()->texture);

        kf_file << ToTum(reader.filenames[id],
                         full_system.all_frame_shells.back()->cam_to_world);
      }

      curr_pose =
          full_system.all_frame_shells.back()->cam_to_world.cast<float>();
      kf_poses.clear();

      for (auto &&fh : full_system.frame_hessians) {
        kf_poses.emplace_back(fh->shell->cam_to_world.cast<float>());
      }

      {
        photo_points.clear();
        homo_points.clear();
        std::vector<cv::Point3f> tracked_points_blue;
        std::vector<cv::Point3f> tracked_points_green;

        for (auto &&fh : full_system.frame_hessians) {
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
        textures[GPUTexture::RGB]->texture->Upload(cv_plot.data, GL_RGB,
                                                   GL_UNSIGNED_BYTE);
      }

      odom_file << ToTum(reader.filenames[id],
                         full_system.all_frame_shells.back()->cam_to_world);

      int num_points = 0, num_homo = 0, num_valid = 0;
      for (auto &&fh : full_system.frame_hessians) {
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
      DLOG(INFO) << "@@@@@@@ ratio: " << point_ratio << "; " << num_valid << ":"
                 << num_points;
    }

    if (pangolin::Pushed(*gui.run_opt)) {
      LOG(INFO) << "SAVE FIGURES";
      cv::Mat cv_out;
      cv::cvtColor(cv_plot, cv_out, CV_RGB2BGR);
      cv::imwrite("/tmp/tracked_points_" + time_str + ".jpg", cv_out);
    }

    if (gui.draw_global_model->Get()) {
      global_model.RenderPointCloud(gui.s_cam.GetProjectionModelViewMatrix(),
                                    gui.draw_normals->Get(),
                                    gui.draw_colors->Get());
    }

    trajectories.emplace_back(full_system.all_frame_shells.back()
                                  ->cam_to_world.translation()
                                  .cast<float>());

    if (gui.draw_trajectory->Get()) {
      glColor3f(1, 0, 0);
      glLineWidth(3);
      pangolin::glDrawLineStrip(trajectories);
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

  delete vbo;

  return 0;
}