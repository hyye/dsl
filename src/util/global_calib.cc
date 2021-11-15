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

#include "util/global_calib.h"

namespace dsl {
int wG[PYR_LEVELS], hG[PYR_LEVELS];
float fxG[PYR_LEVELS], fyG[PYR_LEVELS], cxG[PYR_LEVELS], cyG[PYR_LEVELS];

float fxiG[PYR_LEVELS], fyiG[PYR_LEVELS], cxiG[PYR_LEVELS], cyiG[PYR_LEVELS];

Eigen::Matrix3f KG[PYR_LEVELS], KiG[PYR_LEVELS];

cv::Mat maskG[PYR_LEVELS];

float wM3G;
float hM3G;

float xiG;

float max_half_fovG = M_PI;

bool relocalization::use_superpoint = false;

double CalcMaxHalfFov(int w, double f, double c, double xi) {
  double m = std::max(fabs((w - c) / f), fabs(-c / f));
  m = std::min(m, fabs(tan(asin(1.0 / xi))));
  double mSq = m * m;
  double a0 = mSq + 1;
  double b0 = 2 * xi * mSq;
  double c0 = mSq * xi * xi - 1;
  double delta = b0 * b0 - 4 * a0 * c0;
  if (delta < 0) {
    std::cerr << "CalcMaxHalfFov has negative delta: " << delta << std::endl;
    delta = 0;
  }
  double z = (-b0 + sqrt(delta)) / (2 * a0);
  double x = sqrt(1 - z * z);
  return fabs(atan2(x, z));
}

// deprecated
void SetGlobalCalib(int w, int h, const Eigen::Matrix3f &K) {
  int wlvl = w;
  int hlvl = h;
  pyrLevelsUsed = 1;
  while (wlvl % 2 == 0 && hlvl % 2 == 0 && wlvl * hlvl > 5000 &&
         pyrLevelsUsed < PYR_LEVELS) {
    wlvl /= 2;
    hlvl /= 2;
    pyrLevelsUsed++;
  }
  printf("using pyramid levels 0 to %d. coarsest resolution: %d x %d!\n",
         pyrLevelsUsed - 1, wlvl, hlvl);
  if (wlvl > 100 && hlvl > 100) {
    printf(
        "\n\n===============WARNING!===================\n "
        "using not enough pyramid levels.\n"
        "Consider scaling to a resolution that is a multiple of a power of "
        "2.\n");
  }
  if (pyrLevelsUsed < 3) {
    printf(
        "\n\n===============WARNING!===================\n "
        "I need higher resolution.\n"
        "I will probably segfault.\n");
  }

  wM3G = w - 3;
  hM3G = h - 3;

  wG[0] = w;
  hG[0] = h;
  KG[0] = K;
  fxG[0] = K(0, 0);
  fyG[0] = K(1, 1);
  cxG[0] = K(0, 2);
  cyG[0] = K(1, 2);
  KiG[0] = KG[0].inverse();
  fxiG[0] = KiG[0](0, 0);
  fyiG[0] = KiG[0](1, 1);
  cxiG[0] = KiG[0](0, 2);
  cyiG[0] = KiG[0](1, 2);

  for (int level = 1; level < pyrLevelsUsed; ++level) {
    wG[level] = w >> level;
    hG[level] = h >> level;

    fxG[level] = fxG[level - 1] * 0.5;
    fyG[level] = fyG[level - 1] * 0.5;
    cxG[level] = (cxG[0] + 0.5) / ((int)1 << level) - 0.5;
    cyG[level] = (cyG[0] + 0.5) / ((int)1 << level) - 0.5;

    KG[level] << fxG[level], 0.0, cxG[level], 0.0, fyG[level], cyG[level], 0.0,
        0.0, 1.0;  // synthetic
    KiG[level] = KG[level].inverse();

    fxiG[level] = KiG[level](0, 0);
    fyiG[level] = KiG[level](1, 1);
    cxiG[level] = KiG[level](0, 2);
    cyiG[level] = KiG[level](1, 2);
  }
}

void SetGlobalCalib(int w, int h, const Eigen::Matrix3f &K, float xi) {
  SetGlobalCalib(w, h, K);
  xiG = xi;
  max_half_fovG =
      CalcMaxHalfFov(sqrt(wG[0] * wG[0] + hG[0] * hG[0]), fxG[0], cxG[0], xiG);
}

void SetGlobalMask(std::string mask_file) {
  if (mask_file != "") {
    cv::Mat gray = cv::imread(mask_file, CV_LOAD_IMAGE_GRAYSCALE);
    threshold(gray, maskG[0], 100, 255, cv::THRESH_BINARY);

    for (int level = 1; level < pyrLevelsUsed; ++level) {
      cv::resize(maskG[0], maskG[level], cv::Size(wG[level], hG[level]),
                 cv::INTER_NEAREST);
      //      std::cout << maskG[0].type() << " " << maskG[level].type() <<
      //      std::endl;
    }
  }
}
}  // namespace dsl