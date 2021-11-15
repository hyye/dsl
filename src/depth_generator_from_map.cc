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
#include "depth_generator_from_map.h"
#include "map_render/yaml_loader.h"

using namespace dsl;

DEFINE_string(traj_path, "/mnt/HDD/Datasets/geoloc/trajectory/V101_left_est.tum", "input path");
DEFINE_string(yaml_path, "/mnt/HDD/Datasets/Visual/EuRoC/V1_01_easy/mav0/processed_data/left_pinhole/config.yaml", "input path");
DEFINE_string(output, "/mnt/HDD/Datasets/geoloc/pcds", "output path");
DEFINE_bool(offscreen, false, "offscreen");

std::string ToTum(std::string time, const SE3 &se3) {
  std::stringstream ss;
  Eigen::Quaterniond q = se3.unit_quaternion();
  Vec3 p = se3.translation();
  std::string time_converted;
  if (time.size() >= 18) {
    time_converted = time.substr(0, 10) + "." + time.substr(10, 9);
  } else {
    time_converted = time;
  }
  ss << time_converted << " " << p.x() << " "
     << p.y() << " " << p.z() << " " << q.x() << " " << q.y() << " " << q.z()
     << " " << q.w() << std::endl;
  return ss.str();
}

std::string ToZeroLead(const int value, const unsigned precision)
{
  std::ostringstream oss;
  oss << std::setw(precision) << std::setfill('0') << value;
  return oss.str();
}

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_alsologtostderr = true;

  std::vector<SE3> all_poses;
  SE3 curr_pose;
  int id = 0;

  boost::filesystem::path out_folder(FLAGS_output);
  LOG(INFO) << "Output folder: " << FLAGS_output;

  YamlLoader yaml_loader(FLAGS_yaml_path);
  MapLoader map_loader(yaml_loader.pcd_file, 1,
                       yaml_loader.intensity_scalar, yaml_loader.rgb_scalar, yaml_loader.surfel_size);
  Mat33f K;
  K << yaml_loader.gamma1, 0, yaml_loader.u0, 0, yaml_loader.gamma2,
      yaml_loader.v0, 0, 0, 1;
  Mat33f K_inv = K.inverse();
  SetGlobalCalib(yaml_loader.image_width, yaml_loader.image_height, K,
                 yaml_loader.xi);

  LoadTumToPoses(FLAGS_traj_path, all_poses);

  GUI gui(FLAGS_offscreen, false);
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>
      trajectories;
  float point_ratio = 0;
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>
      photo_points;
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>
      homo_points;
  gui.dataIdx->Meta().range[1] = all_poses.size();

  pangolin::GlBuffer *vbo = map_loader.GetVbo();
  LOG(INFO) << vbo->SizeBytes() << " size bytes";

  GlobalModel global_model;
  global_model.Initialization(*vbo);
  IndexMap index_map;

  gui.FollowAbsPose(all_poses.front().matrix().cast<float>());


  std::map<std::string, GPUTexture *> textures;
  textures[GPUTexture::RGB] = new GPUTexture(wG[0], hG[0], GL_RGBA, GL_RGB,
                                             GL_UNSIGNED_BYTE, true, true);

  while (!pangolin::ShouldQuit()) {
    gui.PreCall();

    if ((!gui.pause->Get() || pangolin::Pushed(*gui.step)) &&
        id < all_poses.size()) {

      LOG(INFO) << "id: " << id;
      curr_pose = all_poses[id];

      pangolin::TypedImage pgl_dist_img;
      index_map.SynthesizeDepth(curr_pose.matrix().cast<float>(), global_model.Model(), 100);
      index_map.DepthTex()->texture->Download(pgl_dist_img);
      cv::Mat dist_img_f = cv::Mat(hG[0], wG[0], CV_32FC1, pgl_dist_img.ptr);

      // cv::namedWindow("dist");
      // cv::imshow("dist", dist_img_f);
      // cv::waitKey(0);
      pcl::PointCloud<pcl::PointXYZ> point_cloud;
      for (int x = 0; x < wG[0]; ++x) {
        for (int y = 0; y < hG[0]; ++y) {
          float dist = dist_img_f.at<float>(y, x);
          if (dist <= 0 || !std::isfinite(dist)) {
            continue;
          }
          Vec3f pix(x, y, 1);
          Vec3f point = K_inv * pix;
          point.normalize();
          point *= dist;
          point_cloud.push_back(pcl::PointXYZ(point.x(), point.y(), point.z()));
        }
      }
      pcl::PCDWriter pcd_writer;
      std::string out_path = (out_folder / (ToZeroLead(id, 5) + ".pcd")).string();
      pcd_writer.writeBinary(out_path, point_cloud);


      if (++id >= all_poses.size()) {
        LOG(INFO) << "finished";
      }

    }

    if (gui.draw_global_model->Get()) {
      global_model.RenderPointCloud(gui.s_cam.GetProjectionModelViewMatrix(),
                                    gui.draw_normals->Get(),
                                    gui.draw_colors->Get());
    }

    trajectories.emplace_back(curr_pose
                                  .translation()
                                  .cast<float>());

    if (gui.draw_trajectory->Get()) {
      glColor3f(1, 0, 0);
      glLineWidth(3);
      pangolin::glDrawLineStrip(trajectories);
      glColor3f(0, 0, 0);
    }

    if (gui.followPose->Get()) {
      gui.FollowAbsPose(curr_pose.matrix().cast<float>());
    }

    gui.dataIdx->operator=(id);
    gui.PostCall();
  }

  return 0;
}