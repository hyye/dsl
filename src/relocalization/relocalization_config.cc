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
// Created by hyye on 7/10/20.
//

#include "relocalization/relocalization_config.h"

namespace dsl::relocalization {

RelocalizationConfig::RelocalizationConfig(std::string filename) : YamlLoader(filename) {
  cv::FileStorage fs_settings(filename, cv::FileStorage::READ);

  if (!fs_settings["best_cliques"].empty())
    fs_settings["best_cliques"] >> best_cliques;
  if (!fs_settings["min_common_weight"].empty())
    fs_settings["min_common_weight"] >> min_common_weight;
  if (!fs_settings["gt_projection"].empty())
    fs_settings["gt_projection"] >> gt_projection;

  if (!fs_settings["max_num_features"].empty())
    fs_settings["max_num_features"] >> max_num_features;

  int tmp_int = 0;

  if (!fs_settings["DescMap.active_search"].empty()) {
    fs_settings["DescMap.active_search"] >> tmp_int;
    desc_map_config.active_search = (tmp_int > 0);
  }
  if (!fs_settings["DescMap.active_search_3d2d"].empty()) {
    fs_settings["DescMap.active_search_3d2d"] >> tmp_int;
    desc_map_config.active_search_3d2d = (tmp_int > 0);
  }
  if (!fs_settings["DescMap.fcheck"].empty()) {
    fs_settings["DescMap.fcheck"] >> tmp_int;
    desc_map_config.fcheck = (tmp_int > 0);
  }
  if (!fs_settings["DescMap.grid_filter"].empty()) {
    fs_settings["DescMap.grid_filter"] >> tmp_int;
    desc_map_config.grid_filter = (tmp_int > 0);
  }
  if (!fs_settings["DescMap.fcheck_cov_kf"].empty()) {
    fs_settings["DescMap.fcheck_cov_kf"] >> tmp_int;
    desc_map_config.fcheck_cov_kf = (tmp_int > 0);
  }
  if (!fs_settings["DescMap.levelsup"].empty()) {
    fs_settings["DescMap.levelsup"] >> tmp_int;
    desc_map_config.levelsup = tmp_int;
  }
  if (!fs_settings["DescMap.enhanced_points"].empty()) {
    fs_settings["DescMap.enhanced_points"] >> tmp_int;
    desc_map_config.enhanced_points = (tmp_int > 0);
  }

  if (!fs_settings["DescMap.use_vlad"].empty()) {
    fs_settings["DescMap.use_vlad"] >> tmp_int;
    desc_map_config.use_vlad = (tmp_int > 0);
  }

  if (!fs_settings["DescMap.use_enhanced_points_in_bf_search"].empty()) {
    fs_settings["DescMap.use_enhanced_points_in_bf_search"] >> tmp_int;
    desc_map_config.use_enhanced_points_in_bf_search = (tmp_int > 0);
  }
  if (!fs_settings["DescMap.use_enhanced_points_in_active_search"].empty()) {
    fs_settings["DescMap.use_enhanced_points_in_active_search"] >> tmp_int;
    desc_map_config.use_enhanced_points_in_active_search = (tmp_int > 0);
  }

  if (!fs_settings["DescMap.use_wide_enhanced_points_in_active_search"].empty()) {
    fs_settings["DescMap.use_wide_enhanced_points_in_active_search"] >> tmp_int;
    desc_map_config.use_wide_enhanced_points_in_active_search = (tmp_int > 0);
  }

  if (!fs_settings["DescMap.use_covisible_kf_points"].empty()) {
    fs_settings["DescMap.use_covisible_kf_points"] >> tmp_int;
    desc_map_config.use_covisible_kf_points = (tmp_int > 0);
  }
  if (!fs_settings["DescMap.use_covisible_kf_points_dbg"].empty()) {
    fs_settings["DescMap.use_covisible_kf_points_dbg"] >> tmp_int;
    desc_map_config.use_covisible_kf_points_dbg = (tmp_int > 0);
  }
  if (!fs_settings["DescMap.use_all_points_in_active_search"].empty()) {
    fs_settings["DescMap.use_all_points_in_active_search"] >> tmp_int;
    desc_map_config.use_all_points_in_active_search = (tmp_int > 0);
  }
  if (!fs_settings["DescMap.use_good_neighbor_points_in_active_search"].empty()) {
    fs_settings["DescMap.use_good_neighbor_points_in_active_search"] >> tmp_int;
    desc_map_config.use_good_neighbor_points_in_active_search = (tmp_int > 0);
  }
  if (!fs_settings["DescMap.min_nGood"].empty()) {
    fs_settings["DescMap.min_nGood"] >> tmp_int;
    desc_map_config.min_nGood = tmp_int;
  }
  if (!fs_settings["DescMap.min_nMatched"].empty()) {
    fs_settings["DescMap.min_nMatched"] >> tmp_int;
    desc_map_config.min_nMatched = tmp_int;
  } else {
    desc_map_config.min_nMatched = desc_map_config.min_nGood;
  }
  if (!fs_settings["DescMap.max_num_kf"].empty()) {
    fs_settings["DescMap.max_num_kf"] >> tmp_int;
    desc_map_config.max_num_kf = tmp_int;
  }
  if (!fs_settings["DescMap.translation_threshold"].empty()) {
    fs_settings["DescMap.translation_threshold"] >> desc_map_config.translation_threshold;
  }

  if (!fs_settings["DescMap.eval_translation_threshold"].empty()) {
    fs_settings["DescMap.eval_translation_threshold"] >> desc_map_config.eval_translation_threshold;
  } else {
    desc_map_config.eval_translation_threshold = desc_map_config.translation_threshold;
  }

  if (!fs_settings["use_superpoint"].empty()) {
    fs_settings["use_superpoint"] >> tmp_int;
    use_superpoint = (tmp_int > 0);
  }

  fs_settings["superpoint_path"] >> superpoint_path;
  if (!fs_settings["pytorch_device"].empty())
    fs_settings["pytorch_device"] >> pytorch_device;
  if (!fs_settings["superpoint_conf_thresh"].empty())
    fs_settings["superpoint_conf_thresh"] >> superpoint_conf_thresh;

  fs_settings["Database.dataset_path"] >> database_dataset_path;
  fs_settings["Database.map_path"] >> database_map_path;
  fs_settings["Database.voc_path"] >> database_voc_path;
  fs_settings["Database.vlad_path"] >> database_vlad_path;

  fs_settings["Query.dataset_path"] >> query_dataset_path;
  fs_settings["Query.gt_traj_path"] >> query_gt_traj_path;
  fs_settings["Query.vlad_path"] >> query_vlad_path;

}

#define STR(x) #x << ':' << ' ' << x

std::string DescMapConfig::Print() {
  std::ostringstream os;
  os << STR(fcheck) << std::endl << STR(grid_filter) << std::endl
     << STR(fcheck_cov_kf) << std::endl << STR(active_search) << std::endl
     << STR(active_search_3d2d) << std::endl << STR(levelsup) << std::endl
     << STR(enhanced_points) << std::endl << STR(use_vlad) << std::endl
     << STR(use_enhanced_points_in_bf_search) << std::endl << STR(use_enhanced_points_in_active_search) << std::endl
     << STR(use_all_points_in_active_search) << std::endl << STR(use_good_neighbor_points_in_active_search) << std::endl
     << STR(use_covisible_kf_points) << std::endl << STR(use_covisible_kf_points_dbg) << std::endl
     << STR(min_nGood) << std::endl << STR(min_nMatched) << std::endl
     << STR(max_num_kf) << std::endl << STR(translation_threshold) << std::endl
     << STR(eval_translation_threshold) << std::endl;
  return os.str();
}

std::string RelocalizationConfig::Print() {
  std::ostringstream os;
  os << "=== desc_map_config ===" << std::endl
     << desc_map_config.Print() << "=== relocalization_config ===" << std::endl
     << STR(superpoint_path) << std::endl << STR(pytorch_device) << std::endl
     << STR(use_superpoint) << std::endl << STR(superpoint_conf_thresh) << std::endl
     << STR(max_num_features) << std::endl << STR(database_dataset_path) << std::endl
     << STR(database_map_path) << std::endl << STR(database_voc_path) << std::endl
     << STR(query_dataset_path) << std::endl << STR(query_gt_traj_path) << std::endl
     << STR(database_vlad_path) << std::endl << STR(query_vlad_path) << std::endl;
  return os.str();
}

} // namespace dsl::relocalization