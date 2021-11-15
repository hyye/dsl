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
#include "relocalization/struct/key_frame_database.h"
#include "relocalization/struct/vlad_database.h"
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

DEFINE_string(yaml_path,
              "/mnt/HDD/Datasets/Visual/EuRoC/V1_02_medium/mav0/processed_data/left_pinhole/relocalization",
              "input yaml path");

// DEFINE_string(path,
//               "/mnt/HDD/Datasets/Visual/EuRoC/V1_02_medium/mav0/processed_data/left_pinhole/relocalization",
//               "input path");
// DEFINE_string(traj_path,
//               "/mnt/HDD/Datasets/Visual/EuRoC/V1_02_medium/mav0/processed_data/left_pinhole/relocalization/V102_left_kf.tum",
//               "traj path");
//
// DEFINE_string(db_path,
//               "/mnt/HDD/Datasets/Visual/EuRoC/V1_02_medium/mav0/processed_data/left_pinhole/relocalization",
//               "db input path");
// DEFINE_string(db_traj_path,
//               "/mnt/HDD/Datasets/Visual/EuRoC/V1_02_medium/mav0/processed_data/left_pinhole/relocalization/V102_left_kf.tum",
//               "db traj path");

// DEFINE_string(voc_path, "/home/hyye/dev_ws/src/dsl/support_files/vocabulary/ORBvoc.bin", "vocabulary path");
DEFINE_bool(offscreen, false, "offscreen");
DEFINE_bool(pause, false, "pause");
DEFINE_bool(vis, true, "visualization");
DEFINE_bool(auto_exit, false, "auto_exit");

// DEFINE_string(load,
//               "/mnt/HDD/Datasets/Visual/EuRoC/V1_02_medium/mav0/processed_data/left_pinhole/relocalization/desc_map.bin",
//               "Map to load");

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_alsologtostderr = true;
  FLAGS_colorlogtostderr = true;

  RelocalizationConfig yaml_loader(FLAGS_yaml_path);
  LOG(INFO) << yaml_loader.Print();

  std::vector<FrameShellWithFn> all_frames_with_fn = LoadTum(yaml_loader.query_gt_traj_path);
  // std::vector<FrameShellWithFn> db_all_frames_with_fn = LoadTum(FLAGS_db_traj_path);

  EurocReader reader(yaml_loader.query_dataset_path);
  EurocReader db_reader(yaml_loader.database_dataset_path);

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
  boost::filesystem::path mask_path = boost::filesystem::path(yaml_loader.query_dataset_path) / "mask.png";
  if (boost::filesystem::exists(mask_path)) {
    SetGlobalMask(mask_path.string());
    LOG(WARNING) << "MASK LOADED: " << mask_path;
  }

  std::unique_ptr<ORBextractor>
      feature_extractor = std::make_unique<ORBextractor>(yaml_loader.max_num_features, 1.2, 8, 20, 7);

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

  LOG(INFO) << "min_common_weight: " << yaml_loader.min_common_weight;

  OrbVocabularyBinary voc;
  voc.loadFromBinaryFile(yaml_loader.database_voc_path);

  GUI gui(FLAGS_offscreen, false);
  float point_ratio = 0;
  gui.dataIdx->Meta().range[1] = all_frames_with_fn.size();
  gui.pause->operator=(FLAGS_pause);

  gui.draw_trajectory->operator=(true);
  // gui.followPose->operator=(true);
  gui.draw_colors->operator=(false);

  pangolin::GlBuffer *vbo = map_loader.GetVbo();
  LOG(INFO) << vbo->SizeBytes() << " size bytes";

  GlobalModel global_model;
  global_model.Initialization(*vbo);

  IndexMap index_map;
  Eigen::Affine3f kf_transform, ex_trans;
  ex_trans = yaml_loader.T_lc;
  kf_transform = yaml_loader.initial_pose * ex_trans;

  std::map < std::string, GPUTexture * > textures;
  textures[GPUTexture::RGB] = new GPUTexture(wG[0], hG[0], GL_RGBA, GL_RGB,
                                             GL_UNSIGNED_BYTE, true, true);
  textures["Model"] = new GPUTexture(wG[0], hG[0], GL_LUMINANCE32F_ARB, GL_LUMINANCE, GL_FLOAT);

  int id = yaml_loader.fast_forward;
  if (id != 0) {
    LOG(WARNING) << "@@@>>> " << id << " <<<@@@";
  }
  SE3 curr_pose = all_frames_with_fn[id].cam_to_world;
  SE3f pnp_pose, raw_pnp_pose;
  gui.FollowAbsPose(curr_pose.matrix().cast<float>());

  DescMap desc_map;
  desc_map.config = yaml_loader.desc_map_config;
  desc_map.SetGlobalVertices(map_loader.vertices, map_loader.vertex_size, map_loader.ptc_size);
  desc_map.SetVocabularyBinary(&voc);

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
  }
  desc_map.SetVLADPath(yaml_loader.database_vlad_path, yaml_loader.query_vlad_path);

  {
    std::vector < Frame * > vpKFs;
    for (auto &&it: desc_map.all_keyframes) {
      vpKFs.push_back(it.second.get());
    }
    desc_map.ComputeFrameNodeIds(vpKFs, desc_map.config.levelsup);

    if (!use_superpoint)
      desc_map.BuildPrioritizedNodes(desc_map.config.levelsup);
    // LOG(INFO) << "vpMPs.size: " << vpMPs.size();
    // LOG(INFO) << "enhanced points: " << desc_map.enhanced_map_points.size();
  }

  // {
  //   std::ofstream ofs;
  //   ofs.open("/tmp/kf_filenames.txt");
  //   for (auto &&kf: desc_map.all_keyframes) {
  //     LOG(INFO) << kf.second->mnId << " " << kf.first;
  //     auto fswfn = db_all_frames_with_fn[kf.second->mnId - 1];
  //     ofs << fswfn.filename << std::endl;
  //   }
  //   ofs.close();
  // }

  typedef typename std::vector<Eigen::Vector3f, Eigen::aligned_allocator < Eigen::Vector3f> > vVec3f;
  vVec3f trajectories, relocalization_traj;
  vVec3f neighbor_points, old_neighbor_points;
  vVec3f associated_points;
  vVec3f desc_points, local_points;
  vVec3f connected_points, rand_points;

  std::vector<Eigen::Matrix4f> covisible_kfs;
  std::vector<Eigen::Matrix4f> detected_kfs;

  double num_frames = 0, num_found = 0, num_wrong = 0;
  bool last_relocOK = true;
  SE3f last_pnp_pose;

  auto t = std::time(nullptr);
  auto tm = *std::localtime(&t);
  std::ostringstream oss;
  oss << std::put_time(&tm, "%y%m%d%H%M%S");
  std::string time_str = oss.str();
  boost::filesystem::path output_folder("/tmp/" + yaml_loader.output_folder);
  if (boost::filesystem::create_directories(output_folder)) {
    LOG(INFO) << "Directory Created: " << output_folder;
  }

  std::ofstream relocal_file;
  relocal_file.open((output_folder / ("relocalization_" + time_str + ".tum")).string());

  cv::Mat dist_img_f(hG[0], wG[0], CV_32FC1);
  dist_img_f = cv::Mat::zeros(hG[0], wG[0], CV_32FC1);

  // TODO: move GUI to another thread
  while (!pangolin::ShouldQuit()) {
    gui.PreCall();

    if ((!gui.pause->Get() || pangolin::Pushed(*gui.step)) && id < all_frames_with_fn.size()) {

      const std::vector<std::string> &reader_fns = reader.filenames;
      const FrameShellWithFn &fs_with_fn = all_frames_with_fn[id];
      int id_in_dataset =
          std::distance(reader_fns.begin(), std::find(reader_fns.begin(), reader_fns.end(), fs_with_fn.filename));

      LOG_ASSERT(fs_with_fn.filename == reader_fns[id_in_dataset]);

      reader.ReadImage(id_in_dataset);
      cv::Mat cv_imgf;
      reader.gray_image.convertTo(cv_imgf, CV_32FC1);

      curr_pose = fs_with_fn.cam_to_world;
      pangolin::TypedImage pgl_dist_img;

      cv::Mat cv_plot;
      cv::cvtColor(reader.gray_image, cv_plot, cv::COLOR_GRAY2RGB);

      timing::Timer extract_timer("feature_extractor");  // around 0.01s

      std::unique_ptr<Frame> keyframe_ptr =
          std::make_unique<Frame>(reader.gray_image, fs_with_fn.filename, feature_extractor.get(), &voc);
      Frame *pF = keyframe_ptr.get();
      pF->SetPose(Converter::toCvMat(curr_pose.inverse()));
      LOG(INFO) << "N: " << pF->N;

      const std::vector<cv::KeyPoint> &keypoints = pF->mvKeysUn;
      const cv::Mat &features = pF->mDescriptors;
      extract_timer.Stop();

      std::vector<std::vector<unsigned int>> neighbor_indices;

      std::vector<PointWithIndex> points_with_indices;
      std::vector < MapPoint * > vpMPMatches;

      {
        timing::Timer desc_map_timer("desc_map");
        bool relocOK = desc_map.Relocalization(pF);
        desc_map_timer.Stop();
        // set desc_map.mvpMapPointMatches to the best match, for visualization only
        // points_with_indices.clear();
        vpMPMatches.clear();
        for (auto &pMP:desc_map.mvpMapPointMatches) {
          if (pMP && !pMP->isBad()) {
            vpMPMatches.push_back(pMP);
            // points_with_indices.emplace_back(0, 0, pMP->idx_in_surfel_map, 0);
          }
        }
        cv::Mat Tcw_pnp = pF->mTcw_pnp;
        if (!Tcw_pnp.empty()) {
          Eigen::Matrix4f pnp_pose_mat;
          cv::cv2eigen(Tcw_pnp.inv(), pnp_pose_mat);
          // index_map.SynthesizeDepth(pnp_pose_mat, global_model.Model(), 100);
          // index_map.DepthTex()->texture->Download(pgl_dist_img);
          // dist_img_f = cv::Mat(hG[0], wG[0], CV_32FC1, pgl_dist_img.ptr);

          if (!relocOK) {
            pF->mTcw_pnp = cv::Mat();
          } else {
            LOG_ASSERT(desc_map.mnMatchKeyFrameDBId != 0);
            Frame *pBestKF = desc_map.all_keyframes[desc_map.mnMatchKeyFrameDBId].get();
            std::vector<std::pair<cv::KeyPoint, cv::KeyPoint>> kpt_pairs;
          }
        }
        covisible_kfs.clear();
        detected_kfs.clear();
        for (auto &pCovKF: desc_map.mvpCovKFs) {
          covisible_kfs.emplace_back(Converter::toSE3Quat(pCovKF->GetPoseInverse()).matrix().cast<float>());
        }
        for (auto &pDetKF: desc_map.mvpDetectedKFs) {
          detected_kfs.emplace_back(Converter::toSE3Quat(pDetKF->GetPoseInverse()).matrix().cast<float>());
        }
      }

      // desc_map.SetPointsAndFrame(points_with_indices, pF, neighbor_indices);
      // Replace the SetPointsAndFrame to obtain desc_map.mvpMapPointMatches

      double r_error = 0, t_error = 0;
      if (!pF->mTcw_pnp.empty()) {
        Eigen::Matrix4f pnp_pose_mat; // TODO: duplicate
        cv::cv2eigen(pF->mTcw_pnp.inv(), pnp_pose_mat);
        Eigen::Matrix3d pnp_Rd = pnp_pose_mat.block(0, 0, 3, 3).cast<double>();
        Eigen::Quaterniond pnp_qd(pnp_Rd);
        Eigen::Quaterniond gt_qd(curr_pose.unit_quaternion());
        r_error = pnp_qd.angularDistance(gt_qd) / M_PI * 180;
        t_error = (pnp_pose_mat.block(0, 3, 3, 1).cast<double>() - curr_pose.translation()).norm();
        LOG(INFO) << " translation error: " << t_error << " rotation error: " << r_error;
      }
      // Visualize matches
      if (FLAGS_vis) {
        timing::Timer vis_matches_timer("vis.matches");
        const std::vector<std::string> &db_fns = db_reader.filenames;
        if (desc_map.mnMatchKeyFrameDBId != 0) {
          db_reader.ReadImage(desc_map.all_keyframes[desc_map.mnMatchKeyFrameDBId]->mTimeStamp);

          cv::Mat out_pair;
          // DrawImagePair(reader.gray_image, db_reader.gray_image, out_pair);

          std::vector<cv::KeyPoint> kpts1, kpts2, kpts_proj;
          std::vector<cv::DMatch> matches1to2;
          Frame *pKF = desc_map.all_keyframes[desc_map.mnMatchKeyFrameDBId].get();

          ConvertMatches(pF, pKF, desc_map.mvpMapPointMatches, kpts1, kpts2, matches1to2);

          if (!pF->mTcw_pnp.empty())
            ConvertMatchesToKpts(pF, desc_map.mvpMapPointMatches, pF->mTcw_pnp, kpts_proj);

          cv::drawMatches(reader.gray_image, kpts1, db_reader.gray_image, kpts2, matches1to2,
                          out_pair);
          for (auto &&kpt_it = kpts1.begin() + matches1to2.size(); kpt_it != kpts1.end(); ++kpt_it) {
            cv::drawMarker(out_pair, kpt_it->pt, cv::Scalar(255, 255, 255), cv::MARKER_CROSS);
          }
          for (auto &&kpt : kpts_proj) {
            cv::drawMarker(out_pair, kpt.pt, cv::Scalar(255, 0, 0), cv::MARKER_SQUARE, 10, 1);
          }
          // for (int mid = 0; mid < desc_map.mvpMPM3D2D.size(); ++mid) {
          //   MapPoint *pMP = desc_map.mvpMPM3D2D[mid];
          //   if (pMP && !pMP->isBad()) {
          //     cv::drawMarker(out_pair, pF->mvKeysUn[mid].pt, cv::Scalar(255, 0, 0), cv::MARKER_CROSS);
          //   }
          // }
          std::string draw_txt;
          if (!pF->mTcw_pnp.empty()) {
            draw_txt = "found " + std::to_string(t_error) + " " + std::to_string(r_error) + " degrees";
          } else {
            draw_txt = "not found";
          }

          AddTextToImage(draw_txt, out_pair, 255, 255, 255);

          cv::namedWindow("pair");
          cv::imshow("pair", out_pair);
          LOG(INFO) << "=======";

          // ConvertMatches(pF, pKF, desc_map.mvpMPM3D2D, kpts1, kpts2, matches1to2);
          //
          // cv::drawMatches(reader.gray_image, kpts1, db_reader.gray_image, kpts2, matches1to2,
          //                 out_pair);
          // for (auto &&kpt_it = kpts1.begin() + matches1to2.size(); kpt_it != kpts1.end(); ++kpt_it) {
          //   cv::drawMarker(out_pair, kpt_it->pt, cv::Scalar(255, 255, 255), cv::MARKER_CROSS);
          // }
          //
          // cv::namedWindow("3d2d");
          // cv::imshow("3d2d", out_pair);

          ConvertMatches(pF, pKF, desc_map.mvpMPRaw, kpts1, kpts2, matches1to2);

          cv::drawMatches(reader.gray_image, kpts1, db_reader.gray_image, kpts2, matches1to2, out_pair);
          for (auto &&kpt_it = kpts1.begin() + matches1to2.size(); kpt_it != kpts1.end(); ++kpt_it) {
            cv::drawMarker(out_pair, kpt_it->pt, cv::Scalar(255, 255, 255), cv::MARKER_CROSS);
          }

          cv::namedWindow("mvpMPRaw");
          cv::imshow("mvpMPRaw", out_pair);

          std::vector<cv::KeyPoint> proj3D2DKF;
          for (auto &&pMP:desc_map.mvpMPM3D2D) {
            if (!pMP || pMP->isBad()) continue;
            float u, v;
            if (pKF->IsInFrustum(pMP->GetWorldPos(), u, v)) {
              proj3D2DKF.emplace_back(u, v, 31, 0);
            }
          }

          // std::sort(proj3D2DKF.begin(), proj3D2DKF.end(), [](const cv::KeyPoint &p1, const cv::KeyPoint &p2) {
          //   if (p1.pt.x < p2.pt.x) return true;
          //   else if (p1.pt.x == p2.pt.x) return p1.pt.y < p2.pt.y;
          //   else return false;
          // });
          //
          // cv::Mat cv_black = cv::Mat::zeros(hG[0], wG[0], CV_8UC3);
          // cv::drawKeypoints(db_reader.gray_image, proj3D2DKF, cv_black);
          // cv::namedWindow("proj3d2d");
          // cv::imshow("proj3d2d", cv_black);
        }
        // visualize matched and covisible images
        std::vector<cv::Mat> vMatchedImg;
        std::vector<cv::Mat> vCovImg;
        // vMatchedImg.push_back(reader.gray_image);
        // for (unsigned long match_frame_id: desc_map.mvCandidateKFId) {
        //   db_reader.ReadImage(desc_map.all_keyframes[match_frame_id]->mTimeStamp);
        //   vMatchedImg.push_back(db_reader.gray_image);
        // }
        // cv::Mat all_matches_img = MakeCanvas(vMatchedImg, 600, 2);
        // cv::namedWindow("all_matches_img");
        // cv::imshow("all_matches_img", all_matches_img);

        // for (Frame *pCovKF: desc_map.mvpCovKFs) {
        //   db_reader.ReadImage(pCovKF->mTimeStamp);
        //   vCovImg.push_back(db_reader.gray_image);
        // }
        // LOG(INFO) << "vMatchedImg: " << vMatchedImg.size() << " vCovImg: " << vCovImg.size();
        // cv::Mat cov_img = MakeCanvas(vCovImg, 600, 2);
        // cv::namedWindow("cov_img");
        // cv::imshow("cov_img", cov_img);

        cv::waitKey(1);
      }

      LOG(INFO) << "vpMPMatches size: " << vpMPMatches.size();

      neighbor_points.clear();
      connected_points.clear();
      rand_points.clear();
      if (!desc_map.mvpLocalMapPoints.empty()) {
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

      cv::Mat Tcw_pnp, Tcw_pnp_raw;
      // Let's validate the selected local map points!
      if (yaml_loader.gt_projection) {
        PnPsolver solver(*pF, desc_map.mvpMapPointMatches);
        solver.SetRansacParameters(0.99, 10, 300, 4, 0.5, 5.991);
        std::vector<bool> vbInliers;
        int nInliers;
        bool bNoMore;
        Tcw_pnp = solver.iterate(200, bNoMore, vbInliers, nInliers);
        Tcw_pnp_raw = Tcw_pnp;
      } else {
        Tcw_pnp = pF->mTcw_pnp;
        Tcw_pnp_raw = pF->mTcw_pnp_raw;
      }

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

      if (!Tcw_pnp.empty()) {
        Eigen::Matrix4f pnp_pose_mat, raw_pnp_pose_mat;
        cv::cv2eigen(Tcw_pnp.inv(), pnp_pose_mat);
        cv::cv2eigen(Tcw_pnp_raw.inv(), raw_pnp_pose_mat);
        pnp_pose.setRotationMatrix(pnp_pose_mat.block<3, 3>(0, 0));
        pnp_pose.translation() = pnp_pose_mat.block<3, 1>(0, 3);
        raw_pnp_pose.setRotationMatrix(raw_pnp_pose_mat.block<3, 3>(0, 0));
        raw_pnp_pose.translation() = raw_pnp_pose_mat.block<3, 1>(0, 3);

        if ((last_pnp_pose.translation() - pnp_pose.translation()).norm() < desc_map.config.translation_threshold) {
          relocalization_traj.emplace_back(pnp_pose.translation().cast<float>());
          num_found += 1;

          relocal_file << ToTum(pF->mTimeStamp, pnp_pose.cast<double>());

          if (t_error > desc_map.config.eval_translation_threshold) {
            num_wrong += 1;
            if (gui.tracking_debug->Get())
              gui.pause->operator=(true);
          }
        }
        last_relocOK = true;
        last_pnp_pose = pnp_pose;
      } else {
        last_relocOK = false;
      }
      num_frames += 1;
      gui.percentageIdx->operator=(num_found / num_frames);
      LOG(INFO) << fmt::format("pose recall: {:.1f}%, current: {}, precision: {:.1f}, num_wrong: {}",
                               num_found / num_frames * 100,
                               !Tcw_pnp.empty() ? "found" : "not found",
                               (num_found - num_wrong) / num_found * 100,
                               num_wrong);
      LOG(INFO) << fmt::format("Detected: {}, Total: {} Wrong: {}",
                               num_found, num_frames, num_wrong);

      desc_points.clear();
      // for (auto &&pwi:points_with_indices) {
      //   cv::Point3f world_pos = desc_map.GetGlobalPoint(pwi.index_in_surfel_map);
      //   desc_points.emplace_back(world_pos.x, world_pos.y, world_pos.z);
      // }
      for (auto &&pMP:vpMPMatches) {
        cv::Point3f world_pos(pMP->GetWorldPos());
        desc_points.emplace_back(world_pos.x, world_pos.y, world_pos.z);
      }
      neighbor_points.clear();
      for (auto &&pMP:desc_map.mvpMapPointMatches) {
        if (!pMP) continue;
        std::vector<unsigned int> nps = pMP->neighbor_points_global_ids;
        for (unsigned int np_idx:nps) {
          cv::Point3f world_pos = desc_map.GetGlobalPoint(np_idx);
          Eigen::Vector3f p_global(world_pos.x, world_pos.y, world_pos.z);
          neighbor_points.push_back(p_global);
        }
      }
      // for (auto &&idx_map_points : desc_map.map_idx_map_points) {
      //   cv::Mat world_pos = idx_map_points.second.front()->GetWorldPos();
      //   Eigen::Vector3f p_global(world_pos.at<float>(0), world_pos.at<float>(1), world_pos.at<float>(2));
      //   desc_points.push_back(p_global);
      // }

      cv::Mat cv_black = cv::Mat::zeros(cv_plot.size(), cv_plot.type());

      cv::drawKeypoints(cv_plot, keypoints, cv_plot,
                        cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
      // cv::drawKeypoints(cv_black, keypoints, cv_black, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

      cv::Mat dist_img_f_vis = (dist_img_f / 5);
      textures[GPUTexture::RGB]->texture->Upload(cv_plot.data, GL_RGB, GL_UNSIGNED_BYTE);
      textures["Model"]->texture->Upload(dist_img_f_vis.data, GL_LUMINANCE, GL_FLOAT);

      if (++id >= all_frames_with_fn.size()) {
        LOG(INFO) << "finished";
        if (FLAGS_auto_exit) break;
      }

      trajectories.emplace_back(curr_pose.translation().cast<float>());

      LOG(INFO) << timing::Timing::Print();
    }

    if (gui.draw_global_model->Get()) {
      global_model.RenderPointCloud(gui.s_cam.GetProjectionModelViewMatrix(),
                                    gui.draw_normals->Get(),
                                    gui.draw_colors->Get());
    }

    gui.DrawWorldPoints(neighbor_points, 1, Eigen::Vector3f(1, 0, 0));
    gui.DrawWorldPoints(local_points, 3, Eigen::Vector3f(1, 1, 1));
    gui.DrawWorldPoints(rand_points, 10, Eigen::Vector3f(1, 1, 0));
    gui.DrawWorldPoints(desc_points, 3, Eigen::Vector3f(0, 1, 0));
    gui.DrawWorldPoints(connected_points, 7, Eigen::Vector3f(0, 1, 1));

    if (gui.draw_trajectory->Get()) {
      glLineWidth(3);
      glColor3f(0, 1, 0);
      pangolin::glDrawLineStrip(trajectories);
      glColor3f(0, 0, 0);

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

    glColor3f(0, 1, 0);
    gui.DrawFrustum(curr_pose.matrix().cast<float>());
    glColor3f(1, 1, 1);

    glColor3f(1, 1, 0);
    gui.DrawFrustum(pnp_pose.matrix());
    glColor3f(1, 1, 1);

    glColor3f(1, 0, 0);
    gui.DrawFrustum(raw_pnp_pose.matrix());
    glColor3f(1, 1, 1);

    glColor3f(0.8, 0.2, 1);
    for (auto &&covisible_kf : covisible_kfs) {
      gui.DrawFrustum(covisible_kf);
    }
    glColor3f(1, 1, 1);

    glColor3f(0.8, 0.6, 0);
    for (auto &&det_kf : detected_kfs) {
      gui.DrawFrustum(det_kf);
    }
    glColor3f(1, 1, 1);

    if (desc_map.mnMatchKeyFrameDBId != 0) {
      Frame *bestFrame = desc_map.all_keyframes[desc_map.mnMatchKeyFrameDBId].get();
      SE3 dbPose = Converter::toSE3Quat(bestFrame->GetPoseInverse());
      glLineWidth(3);
      glColor3f(0, 1, 1);
      gui.DrawFrustum(dbPose.matrix().cast<float>());
      glColor3f(1, 1, 1);
      glLineWidth(1);
    }

    gui.DisplayImg("Model", textures["Model"], true);
    gui.DisplayImg(GPUTexture::RGB, textures[GPUTexture::RGB], true);

    gui.dataIdx->operator=(id);
    gui.PostCall();
  }

  delete vbo;

  return 0;
}