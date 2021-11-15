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

// adapted from DSO

#ifndef DSL_NUM_TYPE_H_
#define DSL_NUM_TYPE_H_

#include <Eigen/Core>
#include "sophus/se3.hpp"
#include "sophus/sim3.hpp"

namespace dsl {

#define MAX_RES_PER_POINT 8
#define NUM_THREADS 6

#define todouble(x) (x).cast<double>()

typedef Sophus::SE3d SE3;
typedef Sophus::SE3f SE3f;
typedef Sophus::Sim3d Sim3;
typedef Sophus::SO3d SO3;
typedef Sophus::SO3f SO3f;

#define CPARS 4 // not always 4

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatXX;
typedef Eigen::Matrix<double, CPARS, CPARS> MatCC;
#define MatToDynamic(x) MatXX(x)

typedef Eigen::Matrix<double, CPARS, 10> MatC10;
typedef Eigen::Matrix<double, 10, 10> Mat1010;
typedef Eigen::Matrix<double, 13, 13> Mat1313;

typedef Eigen::Matrix<double, 8, 10> Mat810;
typedef Eigen::Matrix<double, 8, 3> Mat83;
typedef Eigen::Matrix<double, 6, 6> Mat66;
typedef Eigen::Matrix<double, 5, 3> Mat53;
typedef Eigen::Matrix<double, 4, 3> Mat43;
typedef Eigen::Matrix<double, 4, 2> Mat42;
typedef Eigen::Matrix<double, 3, 3> Mat33;
typedef Eigen::Matrix<double, 2, 2> Mat22;
typedef Eigen::Matrix<double, 8, CPARS> Mat8C;
typedef Eigen::Matrix<double, CPARS, 8> MatC8;
typedef Eigen::Matrix<float, 8, CPARS> Mat8Cf;
typedef Eigen::Matrix<float, CPARS, 8> MatC8f;

typedef Eigen::Matrix<double, 8, 8> Mat88;
typedef Eigen::Matrix<double, 7, 7> Mat77;

typedef Eigen::Matrix<double, CPARS, 1> VecC; // camera parameters
typedef Eigen::Matrix<float, CPARS, 1> VecCf;
typedef Eigen::Matrix<double, 13, 1> Vec13;
typedef Eigen::Matrix<double, 10, 1> Vec10;
typedef Eigen::Matrix<double, 9, 1> Vec9;
typedef Eigen::Matrix<double, 8, 1> Vec8;
typedef Eigen::Matrix<double, 7, 1> Vec7;
typedef Eigen::Matrix<double, 6, 1> Vec6;
typedef Eigen::Matrix<double, 5, 1> Vec5;
typedef Eigen::Matrix<double, 4, 1> Vec4;
typedef Eigen::Matrix<double, 3, 1> Vec3;
typedef Eigen::Matrix<double, 2, 1> Vec2;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VecX;

typedef Eigen::Matrix<float, 3, 3> Mat33f;
typedef Eigen::Matrix<float, 10, 3> Mat103f;
typedef Eigen::Matrix<float, 2, 2> Mat22f;
typedef Eigen::Matrix<float, 3, 1> Vec3f;
typedef Eigen::Matrix<float, 2, 1> Vec2f;
typedef Eigen::Matrix<float, 6, 1> Vec6f;

typedef Eigen::Matrix<double, 4, 9> Mat49;
typedef Eigen::Matrix<double, 8, 9> Mat89;

typedef Eigen::Matrix<double, 9, 4> Mat94;
typedef Eigen::Matrix<double, 9, 8> Mat98;

typedef Eigen::Matrix<double, 8, 1> Mat81;
typedef Eigen::Matrix<double, 1, 8> Mat18;
typedef Eigen::Matrix<double, 9, 1> Mat91;
typedef Eigen::Matrix<double, 1, 9> Mat19;

typedef Eigen::Matrix<double, 8, 4> Mat84;
typedef Eigen::Matrix<double, 4, 8> Mat48;
typedef Eigen::Matrix<double, 4, 4> Mat44;

typedef Eigen::Matrix<float, MAX_RES_PER_POINT, 1> VecNRf;
typedef Eigen::Matrix<double, MAX_RES_PER_POINT, 1> VecNR;
typedef Eigen::Matrix<float, 12, 1> Vec12f;
typedef Eigen::Matrix<float, 1, 8> Mat18f;
typedef Eigen::Matrix<float, 6, 6> Mat66f;
typedef Eigen::Matrix<float, 8, 8> Mat88f;
typedef Eigen::Matrix<float, 8, 4> Mat84f;
typedef Eigen::Matrix<float, 8, 1> Vec8f;
typedef Eigen::Matrix<float, 10, 1> Vec10f;
typedef Eigen::Matrix<float, 6, 6> Mat66f;
typedef Eigen::Matrix<float, 4, 1> Vec4f;
typedef Eigen::Matrix<float, 4, 4> Mat44f;
typedef Eigen::Matrix<float, 12, 12> Mat1212f;
typedef Eigen::Matrix<float, 12, 1> Vec12f;
typedef Eigen::Matrix<float, 13, 13> Mat1313f;
typedef Eigen::Matrix<float, 10, 10> Mat1010f;
typedef Eigen::Matrix<float, 13, 1> Vec13f;
typedef Eigen::Matrix<float, 9, 9> Mat99f;
typedef Eigen::Matrix<float, 9, 1> Vec9f;

typedef Eigen::Matrix<float, 4, 2> Mat42f;
typedef Eigen::Matrix<float, 6, 2> Mat62f;
typedef Eigen::Matrix<float, 1, 2> Mat12f;

typedef Eigen::Matrix<float, Eigen::Dynamic, 1> VecXf;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> MatXXf;

typedef Eigen::Matrix<double, 8 + CPARS + 1, 8 + CPARS + 1>
    MatPCPC; // cpars + xi + ab + r
typedef Eigen::Matrix<float, 8 + CPARS + 1, 8 + CPARS + 1> MatPCPCf;
typedef Eigen::Matrix<double, 8 + CPARS + 1, 1> VecPC;
typedef Eigen::Matrix<float, 8 + CPARS + 1, 1> VecPCf;

typedef Eigen::Matrix<float, 14, 14> Mat1414f;
typedef Eigen::Matrix<float, 14, 1> Vec14f;
typedef Eigen::Matrix<double, 14, 14> Mat1414;
typedef Eigen::Matrix<double, 14, 1> Vec14;

typedef Eigen::Matrix<unsigned char,3,1> Vec3b;

struct AffLight {
  AffLight(double _a, double _b) : a(_a), b(_b){};
  AffLight() : a(0), b(0){};

  // Affine Parameters:
  double a, b; // scalar a and b, just a_t, b_t or a_h, b_h

  /// I_frame = exp(a)*I_global + b. // I_global = exp(-a)*(I_frame - b).
  /// Description of fromToVecExposure,
  /// I_frame = a*I_global + b, I_global = a^-1*(I_frame - b).
  /// \param exposure_h host exposure time
  /// \param exposure_t target exposure time
  /// \param ab_h host ab
  /// \param ab_t target ab
  /// \return [tj*e^aj/(ti*e^ai) bj-tj*e^aj/(ti*e^ai) * bi], j--target, i--host
  static Vec2 FromToVecExposure(float exposure_h, float exposure_t, AffLight ab_h,
                                AffLight ab_t) {
    if (exposure_h == 0 || exposure_t == 0) {
      exposure_t = exposure_h = 1;
      // printf("got exposure value of 0! please choose the correct model.\n");
      // assert(setting_brightnessTransferFunc < 2);
    }

    double a = exp(ab_t.a - ab_h.a) * exposure_t / exposure_h;
    double b = ab_t.b - a * ab_h.b;
    return Vec2(a, b);
  }

  Vec2 Vec() { return Vec2(a, b); }
};

}

#endif // DSL_NUM_TYPE_H_
