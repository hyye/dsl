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

#ifndef DSL_GLOBAL_FUNCS_H_
#define DSL_GLOBAL_FUNCS_H_

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <memory>
#include "num_type.h"

namespace dsl {

// reads interpolated element from a uchar* array
// SSE2 optimization possible
EIGEN_ALWAYS_INLINE float GetInterpolatedElement(const float *const mat,
                                                 const float x, const float y,
                                                 const int width) {
  // stats.num_pixelInterpolations++;

  int ix = (int)x;
  int iy = (int)y;
  float dx = x - ix;
  float dy = y - iy;
  float dxdy = dx * dy;
  const float *bp = mat + ix + iy * width;

  float res = dxdy * bp[1 + width] + (dy - dxdy) * bp[width] +
              (dx - dxdy) * bp[1] + (1 - dx - dy + dxdy) * bp[0];

  return res;
}

EIGEN_ALWAYS_INLINE Eigen::Vector3f GetInterpolatedElement43(
    const Eigen::Vector4f *const mat, const float x, const float y,
    const int width) {
  int ix = (int)x;
  int iy = (int)y;
  float dx = x - ix;
  float dy = y - iy;
  float dxdy = dx * dy;
  const Eigen::Vector4f *bp = mat + ix + iy * width;

  return dxdy * *(const Eigen::Vector3f *)(bp + 1 + width) +
         (dy - dxdy) * *(const Eigen::Vector3f *)(bp + width) +
         (dx - dxdy) * *(const Eigen::Vector3f *)(bp + 1) +
         (1 - dx - dy + dxdy) * *(const Eigen::Vector3f *)(bp);
}

EIGEN_ALWAYS_INLINE Eigen::Vector3f GetInterpolatedElement33(
    const Eigen::Vector3f *const mat, const float x, const float y,
    const int width) {
  int ix = (int)x;
  int iy = (int)y;
  float dx = x - ix;
  float dy = y - iy;
  float dxdy = dx * dy;
  const Eigen::Vector3f *bp = mat + ix + iy * width;

  return dxdy * *(const Eigen::Vector3f *)(bp + 1 + width) +
         (dy - dxdy) * *(const Eigen::Vector3f *)(bp + width) +
         (dx - dxdy) * *(const Eigen::Vector3f *)(bp + 1) +
         (1 - dx - dy + dxdy) * *(const Eigen::Vector3f *)(bp);
}

EIGEN_ALWAYS_INLINE Eigen::Vector3f GetInterpolatedElement33OverAnd(
    const Eigen::Vector3f *const mat, const bool *overMat, const float x,
    const float y, const int width, bool &over_out) {
  int ix = (int)x;
  int iy = (int)y;
  float dx = x - ix;
  float dy = y - iy;
  float dxdy = dx * dy;
  const Eigen::Vector3f *bp = mat + ix + iy * width;

  const bool *bbp = overMat + ix + iy * width;
  over_out = bbp[1 + width] && bbp[1] && bbp[width] && bbp[0];

  return dxdy * *(const Eigen::Vector3f *)(bp + 1 + width) +
         (dy - dxdy) * *(const Eigen::Vector3f *)(bp + width) +
         (dx - dxdy) * *(const Eigen::Vector3f *)(bp + 1) +
         (1 - dx - dy + dxdy) * *(const Eigen::Vector3f *)(bp);
}

EIGEN_ALWAYS_INLINE Eigen::Vector3f GetInterpolatedElement33OverOr(
    const Eigen::Vector3f *const mat, const bool *overMat, const float x,
    const float y, const int width, bool &over_out) {
  int ix = (int)x;
  int iy = (int)y;
  float dx = x - ix;
  float dy = y - iy;
  float dxdy = dx * dy;
  const Eigen::Vector3f *bp = mat + ix + iy * width;

  const bool *bbp = overMat + ix + iy * width;
  over_out = bbp[1 + width] || bbp[1] || bbp[width] || bbp[0];

  return dxdy * *(const Eigen::Vector3f *)(bp + 1 + width) +
         (dy - dxdy) * *(const Eigen::Vector3f *)(bp + width) +
         (dx - dxdy) * *(const Eigen::Vector3f *)(bp + 1) +
         (1 - dx - dy + dxdy) * *(const Eigen::Vector3f *)(bp);
}

EIGEN_ALWAYS_INLINE float GetInterpolatedElement31(
    const Eigen::Vector3f *const mat, const float x, const float y,
    const int width) {
  int ix = (int)x;
  int iy = (int)y;
  float dx = x - ix;
  float dy = y - iy;
  float dxdy = dx * dy;
  const Eigen::Vector3f *bp = mat + ix + iy * width;

  return dxdy * (*(const Eigen::Vector3f *)(bp + 1 + width))[0] +
         (dy - dxdy) * (*(const Eigen::Vector3f *)(bp + width))[0] +
         (dx - dxdy) * (*(const Eigen::Vector3f *)(bp + 1))[0] +
         (1 - dx - dy + dxdy) * (*(const Eigen::Vector3f *)(bp))[0];
}

EIGEN_ALWAYS_INLINE Eigen::Vector3f GetInterpolatedElement13BiLin(
    const float *const mat, const float x, const float y, const int width) {
  int ix = (int)x;
  int iy = (int)y;
  const float *bp = mat + ix + iy * width;

  float tl = *(bp);
  float tr = *(bp + 1);
  float bl = *(bp + width);
  float br = *(bp + width + 1);

  float dx = x - ix;
  float dy = y - iy;
  float topInt = dx * tr + (1 - dx) * tl;
  float botInt = dx * br + (1 - dx) * bl;
  float leftInt = dy * bl + (1 - dy) * tl;
  float rightInt = dy * br + (1 - dy) * tr;

  return Eigen::Vector3f(dx * rightInt + (1 - dx) * leftInt, rightInt - leftInt,
                         botInt - topInt);
}

/// Interpolated grayscale and dx, dy
/// \param mat input Id
/// \param x float x
/// \param y float y
/// \param width
/// \return i, dx, dy
EIGEN_ALWAYS_INLINE Eigen::Vector3f GetInterpolatedElement33BiLin(
    const Eigen::Vector3f *const mat, const float x, const float y,
    const int width) {
  int ix = (int)x;
  int iy = (int)y;
  const Eigen::Vector3f *bp = mat + ix + iy * width;

  float tl = (*(bp))[0];
  float tr = (*(bp + 1))[0];
  float bl = (*(bp + width))[0];
  float br = (*(bp + width + 1))[0];

  float dx = x - ix;
  float dy = y - iy;
  float topInt = dx * tr + (1 - dx) * tl;
  float botInt = dx * br + (1 - dx) * bl;
  float leftInt = dy * bl + (1 - dy) * tl;
  float rightInt = dy * br + (1 - dy) * tr;

  return Eigen::Vector3f(dx * rightInt + (1 - dx) * leftInt, rightInt - leftInt,
                         botInt - topInt);
}

EIGEN_ALWAYS_INLINE float GetInterpolatedElement11Cub(
    const float *const p,
    const float x)  // for x=0, this returns p[1].
{
  return p[1] + 0.5f * x *
                    (p[2] - p[0] +
                     x * (2.0f * p[0] - 5.0f * p[1] + 4.0f * p[2] - p[3] +
                          x * (3.0f * (p[1] - p[2]) + p[3] - p[0])));
}

EIGEN_ALWAYS_INLINE Eigen::Vector2f GetInterpolatedElement12Cub(
    const float *const p,
    const float x)  // for x=0, this returns p[1].
{
  float c1 = 0.5f * (p[2] - p[0]);
  float c2 = p[0] - 2.5f * p[1] + 2 * p[2] - 0.5f * p[3];
  float c3 = 0.5f * (3.0f * (p[1] - p[2]) + p[3] - p[0]);
  float xx = x * x;
  float xxx = xx * x;
  return Eigen::Vector2f(p[1] + x * c1 + xx * c2 + xxx * c3,
                         c1 + x * 2.0f * c2 + xx * 3.0f * c3);
}

EIGEN_ALWAYS_INLINE Eigen::Vector2f GetInterpolatedElement32Cub(
    const Eigen::Vector3f *const p,
    const float x)  // for x=0, this returns p[1].
{
  float c1 = 0.5f * (p[2][0] - p[0][0]);
  float c2 = p[0][0] - 2.5f * p[1][0] + 2 * p[2][0] - 0.5f * p[3][0];
  float c3 = 0.5f * (3.0f * (p[1][0] - p[2][0]) + p[3][0] - p[0][0]);
  float xx = x * x;
  float xxx = xx * x;
  return Eigen::Vector2f(p[1][0] + x * c1 + xx * c2 + xxx * c3,
                         c1 + x * 2.0f * c2 + xx * 3.0f * c3);
}

EIGEN_ALWAYS_INLINE float GetInterpolatedElement11BiCub(const float *const mat,
                                                        const float x,
                                                        const float y,
                                                        const int width) {
  int ix = (int)x;
  int iy = (int)y;
  float dx = x - ix;
  const float *bp = mat + ix + iy * width;

  float val[4];
  val[0] = GetInterpolatedElement11Cub(bp - width - 1, dx);
  val[1] = GetInterpolatedElement11Cub(bp - 1, dx);
  val[2] = GetInterpolatedElement11Cub(bp + width - 1, dx);
  val[3] = GetInterpolatedElement11Cub(bp + 2 * width - 1, dx);

  float dy = y - iy;
  return GetInterpolatedElement11Cub(val, dy);
}
EIGEN_ALWAYS_INLINE Eigen::Vector3f GetInterpolatedElement13BiCub(
    const float *const mat, const float x, const float y, const int width) {
  int ix = (int)x;
  int iy = (int)y;
  float dx = x - ix;
  const float *bp = mat + ix + iy * width;

  float val[4];
  float grad[4];
  Eigen::Vector2f v = GetInterpolatedElement12Cub(bp - width - 1, dx);
  val[0] = v[0];
  grad[0] = v[1];

  v = GetInterpolatedElement12Cub(bp - 1, dx);
  val[1] = v[0];
  grad[1] = v[1];

  v = GetInterpolatedElement12Cub(bp + width - 1, dx);
  val[2] = v[0];
  grad[2] = v[1];

  v = GetInterpolatedElement12Cub(bp + 2 * width - 1, dx);
  val[3] = v[0];
  grad[3] = v[1];

  float dy = y - iy;
  v = GetInterpolatedElement12Cub(val, dy);

  return Eigen::Vector3f(v[0], GetInterpolatedElement11Cub(grad, dy), v[1]);
}

EIGEN_ALWAYS_INLINE Eigen::Vector3f GetInterpolatedElement33BiCub(
    const Eigen::Vector3f *const mat, const float x, const float y,
    const int width) {
  int ix = (int)x;
  int iy = (int)y;
  float dx = x - ix;
  const Eigen::Vector3f *bp = mat + ix + iy * width;

  float val[4];
  float grad[4];
  Eigen::Vector2f v = GetInterpolatedElement32Cub(bp - width - 1, dx);
  val[0] = v[0];
  grad[0] = v[1];

  v = GetInterpolatedElement32Cub(bp - 1, dx);
  val[1] = v[0];
  grad[1] = v[1];

  v = GetInterpolatedElement32Cub(bp + width - 1, dx);
  val[2] = v[0];
  grad[2] = v[1];

  v = GetInterpolatedElement32Cub(bp + 2 * width - 1, dx);
  val[3] = v[0];
  grad[3] = v[1];

  float dy = y - iy;
  v = GetInterpolatedElement12Cub(val, dy);

  return Eigen::Vector3f(v[0], GetInterpolatedElement11Cub(grad, dy), v[1]);
}

EIGEN_ALWAYS_INLINE Eigen::Vector4f GetInterpolatedElement44(
    const Eigen::Vector4f *const mat, const float x, const float y,
    const int width) {
  int ix = (int)x;
  int iy = (int)y;
  float dx = x - ix;
  float dy = y - iy;
  float dxdy = dx * dy;
  const Eigen::Vector4f *bp = mat + ix + iy * width;

  return dxdy * *(bp + 1 + width) + (dy - dxdy) * *(bp + width) +
         (dx - dxdy) * *(bp + 1) + (1 - dx - dy + dxdy) * *(bp);
}

EIGEN_ALWAYS_INLINE Eigen::Vector2f GetInterpolatedElement42(
    const Eigen::Vector4f *const mat, const float x, const float y,
    const int width) {
  int ix = (int)x;
  int iy = (int)y;
  float dx = x - ix;
  float dy = y - iy;
  float dxdy = dx * dy;
  const Eigen::Vector4f *bp = mat + ix + iy * width;

  return dxdy * *(const Eigen::Vector2f *)(bp + 1 + width) +
         (dy - dxdy) * *(const Eigen::Vector2f *)(bp + width) +
         (dx - dxdy) * *(const Eigen::Vector2f *)(bp + 1) +
         (1 - dx - dy + dxdy) * *(const Eigen::Vector2f *)(bp);
}

inline Vec3f MakeRainbowf3F(float id) {
  id *= 1;
  if (id < 0) return Vec3f(1, 1, 1);

  int icP = id;
  float ifP = id - icP;
  icP = icP % 3;

  if (icP == 0) return Vec3f((1 - ifP), ifP, 0);
  if (icP == 1) return Vec3f(0, (1 - ifP), ifP);
  if (icP == 2) return Vec3f(ifP, 0, (1 - ifP));
  assert(false);
  return Vec3f(1, 1, 1);
}

inline Vec3b MakeRainbow3B(float id) {
  id *= 1;
  if (!(id > 0)) return Vec3b(255, 255, 255);

  int icP = id;
  float ifP = id - icP;
  icP = icP % 3;

  if (icP == 0) return Vec3b(255 * (1 - ifP), 255 * ifP, 0);
  if (icP == 1) return Vec3b(0, 255 * (1 - ifP), 255 * ifP);
  if (icP == 2) return Vec3b(255 * ifP, 0, 255 * (1 - ifP));
  return Vec3b(255, 255, 255);
}

inline Vec3b MakeJet3B(float id) {
  if (id <= 0) return Vec3b(128, 0, 0);
  if (id >= 1) return Vec3b(0, 0, 128);

  int icP = (id * 8);
  float ifP = (id * 8) - icP;

  if (icP == 0) return Vec3b(255 * (0.5 + 0.5 * ifP), 0, 0);
  if (icP == 1) return Vec3b(255, 255 * (0.5 * ifP), 0);
  if (icP == 2) return Vec3b(255, 255 * (0.5 + 0.5 * ifP), 0);
  if (icP == 3) return Vec3b(255 * (1 - 0.5 * ifP), 255, 255 * (0.5 * ifP));
  if (icP == 4)
    return Vec3b(255 * (0.5 - 0.5 * ifP), 255, 255 * (0.5 + 0.5 * ifP));
  if (icP == 5) return Vec3b(0, 255 * (1 - 0.5 * ifP), 255);
  if (icP == 6) return Vec3b(0, 255 * (0.5 - 0.5 * ifP), 255);
  if (icP == 7) return Vec3b(0, 0, 255 * (1 - 0.5 * ifP));
  return Vec3b(255, 255, 255);
}

// 0 = red, 1=green, 0.5=yellow.
inline Vec3b MakeRedGreen3B(float val) {
  if (val < 0)
    return Vec3b(0, 0, 255);
  else if (val < 0.5)
    return Vec3b(0, 255 * 2 * val, 255);
  else if (val < 1)
    return Vec3b(0, 255, 255 - 255 * 2 * (val - 0.5));
  else
    return Vec3b(0, 255, 0);
}

template <typename T>
inline void DeleteOut(std::vector<std::unique_ptr<T>> &v, const int i) {
  std::swap(v[i], v.back());
  v.pop_back();
}

template <typename T>
inline void DeleteOutOrder(std::vector<std::unique_ptr<T>> &v,
                           std::unique_ptr<T> &element) {
  int i = -1;
  for (unsigned int k = 0; k < v.size(); k++) {
    if (v[k] == element) {
      i = k;
      break;
    }
  }
  assert(i != -1);

  for (unsigned int k = i + 1; k < v.size(); k++) {
    std::swap(v[k - 1], v[k]);
  }
  v.pop_back();
}

template <typename T>
inline void DeleteOutOrder(std::vector<std::unique_ptr<T>> &v, T *element) {
  int i = -1;
  for (unsigned int k = 0; k < v.size(); k++) {
    if (v[k].get() == element) {
      i = k;
      break;
    }
  }
  assert(i != -1);

  for (unsigned int k = i + 1; k < v.size(); k++) {
    std::swap(v[k - 1], v[k]);
  }
  v.pop_back();
}

template <typename Derived>
inline bool IsNonZero(const Eigen::MatrixBase<Derived> &vec) {
  return fabs(vec.x()) > 1e-6 || fabs(vec.y()) > 1e-6 || fabs(vec.z()) > 1e-6;
}

template <typename Derived>
inline bool IsValid(const Eigen::MatrixBase<Derived> &vec) {
  return std::isfinite(vec.x()) && std::isfinite(vec.y()) &&
         std::isfinite(vec.z()) && IsNonZero(vec);
}

/**
 * Lift from image plane to the unit sphere
 * @tparam T
 * @param host_u
 * @param host_v
 * @param fxi
 * @param fyi
 * @param cxi
 * @param cyi
 * @param xi
 * @return
 */
template <typename T>
inline Eigen::Matrix<T, 3, 1> LiftToSphere(T host_u, T host_v, T fxi, T fyi,
                                           T cxi, T cyi, T xi) {
  Eigen::Matrix<T, 3, 1> p_cam((host_u * fxi + cxi), (host_v * fyi + cyi), 1);
  T d2 = p_cam.x() * p_cam.x() + p_cam.y() * p_cam.y();
  T factor = (xi + sqrt(1 + (1 - xi * xi) * d2)) / (d2 + 1);
  p_cam.z() = 1 - xi / factor;
  p_cam.normalize();
  return p_cam;
}

/**
 * Project space point to image plane
 * @tparam Derived
 * @param p
 * @param fx
 * @param fy
 * @param cx
 * @param cy
 * @param xi
 * @return
 */
template <typename Derived>
inline Eigen::Matrix<typename Derived::Scalar, 3, 1> SpaceToPlane(
    const Eigen::MatrixBase<Derived> &p, typename Derived::Scalar fx,
    typename Derived::Scalar fy, typename Derived::Scalar cx,
    typename Derived::Scalar cy, typename Derived::Scalar xi) {
  typedef typename Derived::Scalar Scalar_t;
  Eigen::Matrix<Scalar_t, 3, 1> ps = p.normalized();
  Scalar_t deno = ps.z() + xi;
  Scalar_t xs = ps.x();
  Scalar_t ys = ps.y();
  Scalar_t xm = xs / deno;
  Scalar_t ym = ys / deno;
  Eigen::Matrix<Scalar_t, 3, 1> plane_point(fx * xm + cx, fy * ym + cy, 1);
  return plane_point;
}

inline bool ValidArea(cv::Mat mask, int u, int v) {
  if (mask.empty()) {
    return true;
  }
  return mask.at<uchar>(v, u) > 0;
}

}  // namespace dsl

#endif  // DSL_GLOBAL_FUNCS_H_
