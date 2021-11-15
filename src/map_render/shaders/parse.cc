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

#include "map_render/shaders/parse.h"

namespace dsl {

Parse::Parse() {}

const Parse &Parse::Get() {
  static const Parse instance;
  return instance;
}

int Parse::Arg(int argc, char **argv, const char *str, std::string &val) const {
  int index = FindArg(argc, argv, str) + 1;

  if (index > 0 && index < argc) {
    val = argv[index];
  }

  return index - 1;
}

int Parse::Arg(int argc, char **argv, const char *str, float &val) const {
  int index = FindArg(argc, argv, str) + 1;

  if (index > 0 && index < argc) {
    val = atof(argv[index]);
  }

  return index - 1;
}

int Parse::Arg(int argc, char **argv, const char *str, int &val) const {
  int index = FindArg(argc, argv, str) + 1;

  if (index > 0 && index < argc) {
    val = atoi(argv[index]);
  }

  return index - 1;
}

std::string Parse::ShaderDir() const {
  std::string currentVal = STR(SHADER_DIR);

  assert(pangolin::FileExists(currentVal) && "Shader directory not found!");

  return currentVal;
}

std::string Parse::BaseDir() const {
  char buf[256];

  int length = readlink("/proc/self/exe", buf, sizeof(buf));

  std::string currentVal;
  currentVal.append((char *)&buf, length);

  currentVal = currentVal.substr(0, currentVal.rfind("/build/"));

  return currentVal;
}

int Parse::FindArg(int argc, char **argv, const char *argument_name) const {
  for (int i = 1; i < argc; ++i) {
    // Search for the string
    if (strcmp(argv[i], argument_name) == 0) {
      return i;
    }
  }
  return -1;
}

}  // namespace dsl