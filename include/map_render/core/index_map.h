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

#ifndef DSL_INDEX_MAP_H_
#define DSL_INDEX_MAP_H_

#include <pangolin/gl/gl.h>
#include <Eigen/LU>
#include "gpu_texture.h"
#include "map_render/shaders/shaders.h"
#include "map_render/shaders/uniform.h"
#include "map_render/shaders/vertex.h"
#include "util/global_calib.h"

namespace dsl {

class IndexMap {
 public:
  IndexMap();
  virtual ~IndexMap();

  void SynthesizeDepth(const Eigen::Matrix4f &pose,
                       const std::pair<GLuint, GLuint> &model,
                       const float depth_cutoff);

  void SynthesizeIntensity(const Eigen::Matrix4f &pose,
                           const std::pair<GLuint, GLuint> &model,
                           const float depth_cutoff);

  void RenderDepth(const float depth_cutoff);

  void RenderDepth(const float depth_cutoff,
                   pangolin::GlTexture *vertex_texture);

  void PredictIndices(const Eigen::Matrix4f &pose,
                      const std::pair<GLuint, GLuint> &model,
                      const float depth_cutoff);

  void PredictGlobalIndices(const Eigen::Matrix4f &pose,
                            const std::pair<GLuint, GLuint> &model,
                            const float depth_cutoff);

  void CombinedPredict(const Eigen::Matrix4f &pose,
                       const std::pair<GLuint, GLuint> &model,
                       const float depth_cutoff);

  GPUTexture *IndexTex() { return &index_texture_; }

  GPUTexture *VertConfTex() { return &vert_conf_texture_; }

  GPUTexture *ColorTimeTex() { return &color_time_texture_; }

  GPUTexture *NormalRadTex() { return &normal_rad_texture_; }

  GPUTexture *VertGlobalTex() { return &vert_global_texture_; }

  GPUTexture *ColorGlobalTex() { return &color_global_texture_; }

  GPUTexture *NormalGlobalTex() { return &normal_global_texture_; }

  GPUTexture *DrawTex() { return &draw_texture_; }

  GPUTexture *IntensityTex() { return &intensity_texture_; }

  GPUTexture *DepthTex() { return &depth_texture_; }

  GPUTexture *ImageTex() { return &image_texture_; }

  GPUTexture *VertexTex() { return &vertex_texture_; }

  GPUTexture *NormalTex() { return &normal_texture_; }

  std::shared_ptr<Shader> index_prog_;
  pangolin::GlFramebuffer index_framebuffer_;
  pangolin::GlRenderBuffer index_renderbuffer_;
  GPUTexture index_texture_;
  GPUTexture vert_conf_texture_;
  GPUTexture color_time_texture_;
  GPUTexture normal_rad_texture_;

  // WARNING wasteful
  std::shared_ptr<Shader> index_global_prog_;
  pangolin::GlFramebuffer index_global_framebuffer_;
  pangolin::GlRenderBuffer index_global_renderbuffer_;
  GPUTexture index_global_texture_;
  GPUTexture vert_global_texture_;
  GPUTexture color_global_texture_;  // NOTE not used
  GPUTexture normal_global_texture_;

  std::shared_ptr<Shader> depth_prog_;
  pangolin::GlFramebuffer depth_framebuffer_;
  pangolin::GlRenderBuffer depth_renderbuffer_;
  GPUTexture depth_texture_;

  std::shared_ptr<Shader> draw_depth_prog_;
  pangolin::GlFramebuffer draw_framebuffer_;
  pangolin::GlRenderBuffer draw_renderbuffer_;
  GPUTexture draw_texture_;

  std::shared_ptr<Shader> intensity_prog_;
  pangolin::GlFramebuffer intensity_framebuffer_;
  pangolin::GlRenderBuffer intensity_renderbuffer_;
  GPUTexture intensity_texture_;

  std::shared_ptr<Shader> combined_prog_;
  pangolin::GlFramebuffer combined_framebuffer_;
  pangolin::GlRenderBuffer combined_renderbuffer_;
  GPUTexture image_texture_;
  GPUTexture vertex_texture_;
  GPUTexture normal_texture_;

  /// NOTE: for debug
  unsigned int count_;
  GLuint count_query_;
  GLuint tbo_;
  ///
};

}

#endif  // DSL_INDEX_MAP_H_
