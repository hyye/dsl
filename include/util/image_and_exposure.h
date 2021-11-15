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

#ifndef DSL_IMAGEANDEXPOSURE_H_
#define DSL_IMAGEANDEXPOSURE_H_

#include "dsl_common.h"

namespace dsl {

struct ImageAndExposure {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  std::vector<float> image; // irradiance. between 0 and 256
  int w, h;                 // width and height;
  double timestamp;
  float exposure_time; // exposure time in ms.
  inline ImageAndExposure(int _w, int _h, double _timestamp = 0)
      : w(_w), h(_h), timestamp(_timestamp) {
    image = std::vector<float>(w * h);
    exposure_time = 1;
  }
  inline ~ImageAndExposure() {}

  inline void CopyMetaTo(ImageAndExposure &other) {
    other.exposure_time = exposure_time;
  }

  inline std::unique_ptr<ImageAndExposure> GetDeepCopy() {
    std::unique_ptr<ImageAndExposure> img =
        std::make_unique<ImageAndExposure>(w, h, timestamp);
    img->exposure_time = exposure_time;
    std::copy(image.begin(), image.end(), std::back_inserter(img->image));
    return img;
  }
};

} // namespace dsl

#endif // DSL_IMAGEANDEXPOSURE_H_
