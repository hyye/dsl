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
// Created by hyye on 6/30/20.
//

#ifndef DSL_DATASET_CONVERTER_H
#define DSL_DATASET_CONVERTER_H

#include <fstream>
#include "util/frame_shell.h"

namespace dsl {

struct FrameShellWithFn : FrameShell {
  std::string filename;
};

// Format 'timestamp tx ty tz qx qy qz qw'
std::vector<FrameShellWithFn> LoadTum(std::string pose_path);

std::string ToTum(std::string time, const SE3 &se3);

}  // namespace dsl

#endif // DSL_DATASET_CONVERTER_H
