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

#ifndef DSL_PARSE_H_
#define DSL_PARSE_H_

#include <dirent.h>

#include <cassert>
#include <string>

#include <unistd.h>

#include <pangolin/utils/file_utils.h>
#include <string.h>

#define XSTR(x) #x
#define STR(x) XSTR(x)

namespace dsl {

class Parse {
 public:
  static const Parse &Get();

  int Arg(int argc, char **argv, const char *str, std::string &val) const;

  int Arg(int argc, char **argv, const char *str, float &val) const;

  int Arg(int argc, char **argv, const char *str, int &val) const;

  std::string ShaderDir() const;

  std::string BaseDir() const;

 private:
  Parse();

  int FindArg(int argc, char **argv, const char *argument_name) const;
};

}

#endif  // DSL_PARSE_H_
