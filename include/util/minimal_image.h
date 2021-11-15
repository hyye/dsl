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
// Created by hyye on 11/7/19.
//

// TODO: redundant?

#ifndef DSL_MINIMAL_IMAGE_H_
#define DSL_MINIMAL_IMAGE_H_

#include "util_common.h"
#include "num_type.h"

namespace dsl {

template <typename T> class MinimalImage {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  int w;
  int h;
  T *data;
  std::vector<T> own_data;

  /*
   * creates minimal image with own memory
   */
  inline MinimalImage(int _w, int _h) : w(_w), h(_h) {
    own_data = std::vector<T>(w * h);
    data = own_data.data();
  }

  /*
   * creates minimal image wrapping around existing memory
   */
  inline MinimalImage(int _w, int _h, T *_data) : w(_w), h(_h) { data = _data; }

  inline ~MinimalImage() {}

  inline std::unique_ptr<MinimalImage> GetClone() {
    std::unique_ptr<MinimalImage> clone = std::make_unique<MinimalImage>(w, h);
    memcpy(clone->data, data, sizeof(T) * w * h);
    return clone;
  }

  inline T &at(int x, int y) { return data[(int)x + ((int)y) * w]; }
  inline T &at(int i) { return data[i]; }

  inline void SetBlack() { memset(data, 0, sizeof(T) * w * h); }

  inline void SetConst(T val) {
    for (int i = 0; i < w * h; i++)
      data[i] = val;
  }

  inline void SetPixel1(const float &u, const float &v, T val) {
    at(u + 0.5f, v + 0.5f) = val;
  }

  inline void SetPixel4(const float &u, const float &v, T val) {
    at(u + 1.0f, v + 1.0f) = val;
    at(u + 1.0f, v) = val;
    at(u, v + 1.0f) = val;
    at(u, v) = val;
  }

  inline void SetPixel9(const int &u, const int &v, T val) {
    at(u + 1, v - 1) = val;
    at(u + 1, v) = val;
    at(u + 1, v + 1) = val;
    at(u, v - 1) = val;
    at(u, v) = val;
    at(u, v + 1) = val;
    at(u - 1, v - 1) = val;
    at(u - 1, v) = val;
    at(u - 1, v + 1) = val;
  }

  inline void SetPixelCirc(const int &u, const int &v, T val) {
    for (int i = -3; i <= 3; i++) {
      at(u + 3, v + i) = val;
      at(u - 3, v + i) = val;
      at(u + 2, v + i) = val;
      at(u - 2, v + i) = val;

      at(u + i, v - 3) = val;
      at(u + i, v + 3) = val;
      at(u + i, v - 2) = val;
      at(u + i, v + 2) = val;
    }
  }

private:
};

typedef Eigen::Matrix<unsigned char, 3, 1> Vec3b;
typedef MinimalImage<float> MinimalImageF;
typedef MinimalImage<Vec3f> MinimalImageF3;
typedef MinimalImage<unsigned char> MinimalImageB;
typedef MinimalImage<Vec3b> MinimalImageB3;
typedef MinimalImage<unsigned short> MinimalImageB16;

} // namespace dsl

#endif // DSL_MINIMAL_IMAGE_H_
