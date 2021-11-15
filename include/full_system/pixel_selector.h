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
// Created by hyye on 11/11/19.
//

#ifndef DSL_PIXEL_SELECTOR_H_
#define DSL_PIXEL_SELECTOR_H_

#include "hessian_blocks.h"
#include "util/global_calib.h"
#include "util/global_funcs.h"
#include "util/num_type.h"

namespace dsl {

enum PixelSelectorStatus { PIXSEL_VOID = 0, PIXSEL_1, PIXSEL_2, PIXSEL_3 };

class PixelSelector {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  /**
   *
   * @param fh input frame_hessian
   * @param map_out the PixelSelectorStatus, selected in three levels as n[0],
   * n[1], n[2], (n2,n3,n4)
   * @param density
   * @param recursions_left
   * @param th_factor
   * @return
   */
  int MakeMaps(const FrameHessian& fh, std::vector<float>& map_out,
               float density, int recursions_left = 1, float th_factor = 1);

  PixelSelector(int w, int h);
  ~PixelSelector() {}
  int current_potential;

  void MakeHists(const FrameHessian& fh);

 private:
  Eigen::Vector3i Select(const FrameHessian& fh, std::vector<float>& map_out,
                         int pot, float th_factor = 1);

  std::vector<unsigned char> random_pattern;

  std::vector<int> grad_hist;
  std::vector<float> ths;
  std::vector<float> ths_smoothed;
  int ths_step;
  const FrameHessian* grad_hist_frame;
};

}  // namespace dsl

#endif  // DSL_PIXEL_SELECTOR_H_
