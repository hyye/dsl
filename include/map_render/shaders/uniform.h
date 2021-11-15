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
// Created by hyye on 12/27/19.
//

#ifndef DSL_UNIFORM_H_
#define DSL_UNIFORM_H_

#include <Eigen/Core>
#include <string>

class Uniform {
 public:
  Uniform(const std::string& id, const int& v) : id(id), i(v), t(INT) {}

  Uniform(const std::string& id, const float& v) : id(id), f(v), t(FLOAT) {}

  Uniform(const std::string& id, const Eigen::Vector2f& v)
      : id(id), v2(v), t(VEC2) {}

  Uniform(const std::string& id, const Eigen::Vector3f& v)
      : id(id), v3(v), t(VEC3) {}

  Uniform(const std::string& id, const Eigen::Vector4f& v)
      : id(id), v4(v), t(VEC4) {}

  Uniform(const std::string& id, const Eigen::Matrix4f& v)
      : id(id), m4(v), t(MAT4) {}

  std::string id;

  int i;
  float f;
  Eigen::Vector2f v2;
  Eigen::Vector3f v3;
  Eigen::Vector4f v4;
  Eigen::Matrix4f m4;

  enum Type { INT, FLOAT, VEC2, VEC3, VEC4, MAT4, NONE };

  Type t;
};

#endif  // DSL_UNIFORM_H_
