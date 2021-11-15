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
// Created by hyye on 7/7/20.
//

#ifndef DSL_DESC_MAP_VIEWER_H
#define DSL_DESC_MAP_VIEWER_H

#include "map_render/gui.h"
#include "map_render/core/global_model.h"

namespace dsl::relocalization {

// TODO:
class DescMapViewer : public GUI {
 public:
  DescMapViewer();
  void Run();
  void DrawAll();
  bool GetPauseFlag();
  bool GetStepFlag();
  bool CheckFinish();
  bool pause_flag = false, step_flag = false;
  bool finish_flag = false;
  GlobalModel *global_model;
  std::mutex gui_mutex;
};

} // namespace dsl::relocalization



#endif // DSL_DESC_MAP_VIEWER_H
