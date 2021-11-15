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

#ifndef DSL_FEEDBACK_BUFFER_H_
#define DSL_FEEDBACK_BUFFER_H_

#include <pangolin/display/opengl_render_state.h>
#include <pangolin/gl/gl.h>
#include "map_render/shaders/shaders.h"
#include "map_render/shaders/uniform.h"
#include "map_render/shaders/vertex.h"
#include "util/global_calib.h"

namespace dsl {

class FeedbackBuffer {
 public:
  FeedbackBuffer(std::shared_ptr<Shader> program);
  virtual ~FeedbackBuffer();

  std::shared_ptr<Shader> program;

  void compute(pangolin::GlTexture *color, pangolin::GlTexture *depth,
               const int &time, const float depthCutoff);

  void render(pangolin::OpenGlMatrix mvp, const Eigen::Matrix4f &pose,
              const bool drawNormals, const bool drawColors);

  static const std::string RAW, FILTERED, PREDICT;

  GLuint vbo;
  GLuint fid;

 private:
  std::shared_ptr<Shader> drawProgram;
  GLuint uvo;
  GLuint countQuery;
  const int bufferSize;
  unsigned int count;
};

}  // namespace dsl

#endif  // DSL_FEEDBACK_BUFFER_H_
