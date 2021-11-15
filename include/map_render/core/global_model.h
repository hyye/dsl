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

#ifndef DSL_GLOBAL_MODEL_H_
#define DSL_GLOBAL_MODEL_H_

#include <pangolin/gl/gl.h>
#include "map_render/shaders/shaders.h"
#include "map_render/shaders/uniform.h"
#include "map_render/shaders/feedback_buffer.h"
#include "gpu_texture.h"

namespace dsl {

class GlobalModel {

 public:
  GlobalModel();
  virtual ~GlobalModel();

  static const int TEXTURE_DIMENSION;
  static const int MAX_VERTICES;

  void Initialization(const GLuint &in_vbo, const GLuint &in_fid);
  void Initialization(const pangolin::GlBuffer &vbo);
  void RenderPointCloud(pangolin::OpenGlMatrix mvp,
                        const bool draw_normals = false,
                        const bool draw_colors = true);
  void Fuse(const Eigen::Matrix4f &pose,
            GPUTexture *rgb,
            GPUTexture *index_map,
            GPUTexture *vert_conf_map,
            GPUTexture *color_time_map,
            GPUTexture *norm_rad_map,
            const float depth_cutoff);
  const std::pair<GLuint, GLuint> &Model();

  unsigned int GetCount();

  Eigen::Vector4f *DownloadMap();

 private:
  const int buffer_size_;

  //First is the vbo, second is the fid
  std::pair<GLuint, GLuint> *vbos_;
  GLuint *vaos_;
  int target_, render_;
  unsigned int count_;
  GLuint count_query_;
  std::shared_ptr<Shader> init_prog_;
  std::shared_ptr<Shader> draw_surfel_prog_;
  std::shared_ptr<Shader> data_prog_;
  std::shared_ptr<Shader> update_prog_;

  GLuint new_unstable_vbo_, new_unstable_fid_;

  pangolin::GlRenderBuffer render_buffer_;
  pangolin::GlFramebuffer framebuffer_;

  GPUTexture update_map_verts_confs_;
  GPUTexture update_map_colors_time_;
  GPUTexture update_map_norms_radii_;

  GLuint uvo_;
  int uv_size_;

};

}

#endif  // DSL_GLOBAL_MODEL_H_
