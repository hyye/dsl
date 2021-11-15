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
// Created by hyye on 11/6/19.
//

#ifndef DSL_GLOBALCALIB_H_
#define DSL_GLOBALCALIB_H_

#include "settings.h"
#include "num_type.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace dsl {

extern int wG[PYR_LEVELS], hG[PYR_LEVELS];
extern float fxG[PYR_LEVELS], fyG[PYR_LEVELS], cxG[PYR_LEVELS], cyG[PYR_LEVELS];

extern float fxiG[PYR_LEVELS], fyiG[PYR_LEVELS], cxiG[PYR_LEVELS],
    cyiG[PYR_LEVELS];

extern Eigen::Matrix3f KG[PYR_LEVELS], KiG[PYR_LEVELS];

extern cv::Mat maskG[PYR_LEVELS];

extern float wM3G;
extern float hM3G;

extern float xiG;

extern float max_half_fovG;

void SetGlobalCalib(int w, int h, const Eigen::Matrix3f &K);
void SetGlobalCalib(int w, int h, const Eigen::Matrix3f &K, float xi);
void SetGlobalMask(std::string mask_file);

namespace relocalization {
// WARNING: config for relocalization only
extern bool use_superpoint;
}

} // namespace dsl

#endif // DSL_GLOBALCALIB_H_
