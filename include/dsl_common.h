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
// Created by hyye on 11/5/19.
//

#ifndef DSL_DSL_COMMON_H_
#define DSL_DSL_COMMON_H_

#include <iomanip>
#include <fstream>
#include "util/settings.h"
#include "util/num_type.h"
#include "util/global_calib.h"
#include "util/global_funcs.h"
#include "util/index_thread_reduce.h"
#include "util/minimal_image.h"
#include "util/image_and_exposure.h"
#include "util/frame_shell.h"
#include "util/timing.h"
#include "optimization/parameter_map.h"

// really want to get rid of these
#define SCALE_IDIST 1.0f  // scales internal value to idist.
#define SCALE_XI_ROT 1.0f
#define SCALE_XI_TRANS 0.5f
#define SCALE_F 50.0f
#define SCALE_C 50.0f
#define SCALE_W 1.0f
#define SCALE_A 10.0f
#define SCALE_B 1000.0f

namespace dsl {

struct ImmaturePoint;
struct FrameHessian;
struct PointHessian;
struct PointFrameResidual;
struct EfFrame;
struct EfPoint;
struct EfResidual;
class EnergyFunction;

struct CalibHessian;

enum ResLocation { ACTIVE = 0, LINEARIZED, MARGINALIZED, NONE };
enum ResState { IN = 0, OOB, OUTLIER };
// TODO: SCALES

}



#endif // DSL_DSL_COMMON_H_
