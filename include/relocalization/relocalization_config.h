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

#ifndef DSL_RELOCALIZATION_CONFIG_H
#define DSL_RELOCALIZATION_CONFIG_H

#include "map_render/yaml_loader.h"

namespace dsl::relocalization {

class DescMapConfig {
 public:
  bool fcheck = false;
  bool grid_filter = false;
  bool fcheck_cov_kf = false;
  bool active_search = false;
  bool active_search_3d2d = false;

  int levelsup = 2;
  bool enhanced_points = false;
  bool use_vlad = true;
  bool use_enhanced_points_in_bf_search = false;
  bool use_enhanced_points_in_active_search = false;
  bool use_wide_enhanced_points_in_active_search = false;
  bool use_all_points_in_active_search = false;
  bool use_good_neighbor_points_in_active_search = true;
  bool use_covisible_kf_points = true;
  bool use_covisible_kf_points_dbg = true;
  int min_nGood = 50;
  int min_nMatched = 50;
  int max_num_kf = 3;

  double translation_threshold = 0.3;
  double eval_translation_threshold = 0.3;

  std::string Print();
};

class RelocalizationConfig : public YamlLoader {
 public:
  RelocalizationConfig(std::string filename);
  int max_num_features = 1000;
  int gt_projection = 0;
  int best_cliques = 3;
  int min_common_weight = 2;
  int num_features = 500;
  double scale_factor = 1.2;
  int num_levels = 5;

  bool use_superpoint = false;
  std::string superpoint_path;
  std::string pytorch_device = "CUDA";
  float superpoint_conf_thresh = 0.007;

  std::string database_dataset_path;
  std::string database_map_path;
  std::string database_voc_path;

  std::string query_dataset_path;
  std::string query_gt_traj_path;

  std::string database_vlad_path;
  std::string query_vlad_path;

  DescMapConfig desc_map_config;

  std::string Print();
};

} // namespace dsl::relocalization

#endif // DSL_RELOCALIZATION_CONFIG_H
