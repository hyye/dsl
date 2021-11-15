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

#include "map_render/core/global_model.h"

#include "map_render/core/gpu_texture.h"

#include <Eigen/StdVector>

namespace dsl {

const int GlobalModel::TEXTURE_DIMENSION = 3072;
const int GlobalModel::MAX_VERTICES =
    GlobalModel::TEXTURE_DIMENSION * GlobalModel::TEXTURE_DIMENSION;

GlobalModel::GlobalModel()
    : target_(0),
      render_(1),
      buffer_size_(MAX_VERTICES * Vertex::SIZE),
      render_buffer_(TEXTURE_DIMENSION, TEXTURE_DIMENSION),
      update_map_verts_confs_(TEXTURE_DIMENSION, TEXTURE_DIMENSION, GL_RGBA32F,
                              GL_LUMINANCE, GL_FLOAT),
      update_map_colors_time_(TEXTURE_DIMENSION, TEXTURE_DIMENSION, GL_RGBA32F,
                              GL_LUMINANCE, GL_FLOAT),
      update_map_norms_radii_(TEXTURE_DIMENSION, TEXTURE_DIMENSION, GL_RGBA32F,
                              GL_LUMINANCE, GL_FLOAT),
      init_prog_(LoadProgramFromFile("init_unstable.vert")),
      draw_surfel_prog_(LoadProgramFromFile("draw_surface.vert",
                                            "draw_surface.frag",
                                            "draw_surface.geom")),
      data_prog_(LoadProgramFromFile("data.vert", "data.frag", "data.geom")),
      update_prog_(LoadProgramFromFile("update_with_pose.vert")) {
  vbos_ = new std::pair<GLuint, GLuint>[2];
  vaos_ = new GLuint[2];
  glGenVertexArrays(1, &vaos_[0]);
  glGenVertexArrays(1, &vaos_[1]);

  float *vertices = new float[buffer_size_ / sizeof(float)];

  memset(&vertices[0], 0, buffer_size_);

  glGenTransformFeedbacks(1, &vbos_[0].second);
  glGenBuffers(1, &vbos_[0].first);
  glBindBuffer(GL_ARRAY_BUFFER, vbos_[0].first);
  glBufferData(GL_ARRAY_BUFFER, buffer_size_, &vertices[0], GL_STREAM_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  glGenTransformFeedbacks(1, &vbos_[1].second);
  glGenBuffers(1, &vbos_[1].first);
  glBindBuffer(GL_ARRAY_BUFFER, vbos_[1].first);
  glBufferData(GL_ARRAY_BUFFER, buffer_size_, &vertices[0], GL_STREAM_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  delete[] vertices;

  vertices = new float[wG[0] * hG[0] * Vertex::SIZE];

  memset(&vertices[0], 0, wG[0] * hG[0] * Vertex::SIZE);

  glGenTransformFeedbacks(1, &new_unstable_fid_);
  glGenBuffers(1, &new_unstable_vbo_);
  glBindBuffer(GL_ARRAY_BUFFER, new_unstable_vbo_);
  glBufferData(GL_ARRAY_BUFFER, wG[0] * hG[0] * Vertex::SIZE, &vertices[0],
               GL_STREAM_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  delete[] vertices;

  std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f>> uv;

  for (int i = 0; i < wG[0]; i++) {
    for (int j = 0; j < hG[0]; j++) {
      uv.push_back(Eigen::Vector2f(
          ((float)i / (float)wG[0]) + 1.0 / (2 * (float)wG[0]),
          ((float)j / (float)hG[0]) + 1.0 / (2 * (float)hG[0])));
    }
  }

  uv_size_ = uv.size();

  glGenBuffers(1, &uvo_);
  glBindBuffer(GL_ARRAY_BUFFER, uvo_);
  glBufferData(GL_ARRAY_BUFFER, uv_size_ * sizeof(Eigen::Vector2f), &uv[0],
               GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  framebuffer_.AttachColour(*update_map_verts_confs_.texture);
  framebuffer_.AttachColour(*update_map_colors_time_.texture);
  framebuffer_.AttachColour(*update_map_norms_radii_.texture);
  framebuffer_.AttachDepth(render_buffer_);

  {
    data_prog_->Bind();

    int dataUpdate[3] = {
        glGetVaryingLocationNV(data_prog_->ProgramId(), "vPosition0"),
        glGetVaryingLocationNV(data_prog_->ProgramId(), "vColor0"),
        glGetVaryingLocationNV(data_prog_->ProgramId(), "vNormRad0"),
    };

    glTransformFeedbackVaryingsNV(data_prog_->ProgramId(), 3, dataUpdate,
                                  GL_INTERLEAVED_ATTRIBS);

    data_prog_->Unbind();
  }

  {
    update_prog_->Bind();

    int locUpdate[3] = {
        glGetVaryingLocationNV(update_prog_->ProgramId(), "vPosition0"),
        glGetVaryingLocationNV(update_prog_->ProgramId(), "vColor0"),
        glGetVaryingLocationNV(update_prog_->ProgramId(), "vNormRad0"),
    };
    glTransformFeedbackVaryingsNV(update_prog_->ProgramId(), 3, locUpdate,
                                  GL_INTERLEAVED_ATTRIBS);

    update_prog_->Unbind();
  }

  {
    init_prog_->Bind();

    int locInit[3] = {
        glGetVaryingLocationNV(init_prog_->ProgramId(), "vPosition0"),
        glGetVaryingLocationNV(init_prog_->ProgramId(), "vColor0"),
        glGetVaryingLocationNV(init_prog_->ProgramId(), "vNormRad0"),
    };

    glTransformFeedbackVaryingsNV(init_prog_->ProgramId(), 3, locInit,
                                  GL_INTERLEAVED_ATTRIBS);

    glGenQueries(1, &count_query_);

    // Empty both transform feedbacks
    glEnable(GL_RASTERIZER_DISCARD);

    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, vbos_[0].second);

    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, vbos_[0].first);

    glBeginTransformFeedback(GL_POINTS);

    glDrawArrays(GL_POINTS, 0, 0);

    glEndTransformFeedback();

    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);

    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, vbos_[1].second);

    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, vbos_[1].first);

    glBeginTransformFeedback(GL_POINTS);

    glDrawArrays(GL_POINTS, 0, 0);

    glEndTransformFeedback();

    glDisable(GL_RASTERIZER_DISCARD);

    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);

    init_prog_->Unbind();
  }
}

GlobalModel::~GlobalModel() {
  glDeleteBuffers(1, &vbos_[0].first);
  glDeleteTransformFeedbacks(1, &vbos_[0].second);

  glDeleteBuffers(1, &vbos_[1].first);
  glDeleteTransformFeedbacks(1, &vbos_[1].second);

  glDeleteVertexArrays(1, &vaos_[0]);
  glDeleteVertexArrays(1, &vaos_[1]);

  glDeleteTransformFeedbacks(1, &new_unstable_fid_);
  glDeleteBuffers(1, &new_unstable_vbo_);

  glDeleteQueries(1, &count_query_);
  delete[] vbos_;
  delete[] vaos_;
}

// NOTE: deprecated
void GlobalModel::Initialization(const GLuint &in_vbo, const GLuint &in_fid) {
  init_prog_->Bind();

  //  glBindVertexArray(vaos_[target_]);

  glBindBuffer(GL_ARRAY_BUFFER, in_vbo);

  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, 0);

  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE,
                        reinterpret_cast<GLvoid *>(sizeof(Eigen::Vector4f)));

  glEnableVertexAttribArray(2);
  glVertexAttribPointer(
      2, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE,
      reinterpret_cast<GLvoid *>(sizeof(Eigen::Vector4f) * 2));
  glEnable(GL_RASTERIZER_DISCARD);

  glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, vbos_[target_].second);

  glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, vbos_[target_].first);

  glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, count_query_);

  glBeginTransformFeedback(GL_POINTS);

  // It's ok to use either fid because both raw and filtered have the same
  // amount of vertices
  glDrawTransformFeedback(GL_POINTS, in_fid);

  glEndTransformFeedback();

  glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);

  glGetQueryObjectuiv(count_query_, GL_QUERY_RESULT, &count_);

#ifdef DEBUG_INIT_PROG
  LOG(INFO) << "count_: " << count_;
#endif

  glDisable(GL_RASTERIZER_DISCARD);
  glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);

  glDisableVertexAttribArray(0);
  // NOTE: for test
  glDisableVertexAttribArray(1);
  glDisableVertexAttribArray(2);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  init_prog_->Unbind();

  //  glBindVertexArray(0);

  glFinish();
}

void GlobalModel::Initialization(const pangolin::GlBuffer &vbo) {
  init_prog_->Bind();

  glBindBuffer(GL_ARRAY_BUFFER, vbo.bo);

  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, 0);

  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE,
                        reinterpret_cast<GLvoid *>(sizeof(Eigen::Vector4f)));

  glEnableVertexAttribArray(2);
  glVertexAttribPointer(
      2, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE,
      reinterpret_cast<GLvoid *>(sizeof(Eigen::Vector4f) * 2));
  {
    glEnable(GL_RASTERIZER_DISCARD);

    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, vbos_[target_].second);

    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, vbos_[target_].first);

    glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, count_query_);

    glBeginTransformFeedback(GL_POINTS);

    glDrawArrays(GL_POINTS, 0, vbo.num_elements);

    glEndTransformFeedback();

    glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);

    glGetQueryObjectuiv(count_query_, GL_QUERY_RESULT, &count_);

    glDisable(GL_RASTERIZER_DISCARD);
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);
  }

  {  // FIXME: duplicate
    glEnable(GL_RASTERIZER_DISCARD);

    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, vbos_[render_].second);

    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, vbos_[render_].first);

    glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, count_query_);

    glBeginTransformFeedback(GL_POINTS);

    glDrawArrays(GL_POINTS, 0, vbo.num_elements);

    glEndTransformFeedback();

    glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);

    glGetQueryObjectuiv(count_query_, GL_QUERY_RESULT, &count_);

    glDisable(GL_RASTERIZER_DISCARD);
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);
  }

  glDisableVertexAttribArray(0);
  glDisableVertexAttribArray(1);
  glDisableVertexAttribArray(2);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  init_prog_->Unbind();

  glFinish();
}

const std::pair<GLuint, GLuint> &GlobalModel::Model() { return vbos_[target_]; }

unsigned int GlobalModel::GetCount() { return count_; }

void GlobalModel::Fuse(const Eigen::Matrix4f &pose, GPUTexture *rgb,
                       GPUTexture *index_map, GPUTexture *vert_conf_map,
                       GPUTexture *color_time_map, GPUTexture *norm_rad_map,
                       const float depth_cutoff) {
  framebuffer_.Bind();

  glPushAttrib(GL_VIEWPORT_BIT);

  glViewport(0, 0, render_buffer_.width, render_buffer_.height);

  glClearColor(0, 0, 0, 0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  data_prog_->Bind();

  data_prog_->SetUniform(Uniform("cSampler", 0));
  data_prog_->SetUniform(Uniform("indexSampler", 1));
  data_prog_->SetUniform(Uniform("vertConfSampler", 2));
  data_prog_->SetUniform(Uniform("colorTimeSampler", 3));
  data_prog_->SetUniform(Uniform("normRadSampler", 4));

  data_prog_->SetUniform(Uniform(
      "cam", Eigen::Vector4f(cxG[0], cyG[0], 1.0 / fxG[0], 1.0 / fyG[0])));
  data_prog_->SetUniform(Uniform("cols", (float)wG[0]));
  data_prog_->SetUniform(Uniform("rows", (float)hG[0]));

  data_prog_->SetUniform(Uniform("texDim", (float)TEXTURE_DIMENSION));
  data_prog_->SetUniform(Uniform("pose", pose));
  data_prog_->SetUniform(Uniform("maxDepth", depth_cutoff));

  glEnableVertexAttribArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, uvo_);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);

  glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, new_unstable_fid_);

  glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, new_unstable_vbo_);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, rgb->texture->tid);

  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, index_map->texture->tid);

  glActiveTexture(GL_TEXTURE2);
  glBindTexture(GL_TEXTURE_2D, vert_conf_map->texture->tid);

  glActiveTexture(GL_TEXTURE3);
  glBindTexture(GL_TEXTURE_2D, color_time_map->texture->tid);

  glActiveTexture(GL_TEXTURE4);
  glBindTexture(GL_TEXTURE_2D, norm_rad_map->texture->tid);

#ifdef DEBUG_UPDATE_PROG
  glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, count_query_);
#endif

  glBeginTransformFeedback(GL_POINTS);

  glDrawArrays(GL_POINTS, 0, uv_size_);

  glEndTransformFeedback();

#ifdef DEBUG_UPDATE_PROG
  glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);

  unsigned int data_count;
  glGetQueryObjectuiv(count_query_, GL_QUERY_RESULT, &data_count);

  LOG(INFO) << "data count: " << data_count;
#endif

  framebuffer_.Unbind();

  glBindTexture(GL_TEXTURE_2D, 0);

  glActiveTexture(GL_TEXTURE0);

  glDisableVertexAttribArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);

  data_prog_->Unbind();

  Eigen::Matrix4f pose_inv = pose.inverse();

  glPopAttrib();
  glFinish();

  update_prog_->Bind();

  update_prog_->SetUniform(Uniform("vertSamp", 0));
  update_prog_->SetUniform(Uniform("colorSamp", 1));
  update_prog_->SetUniform(Uniform("normSamp", 2));
  update_prog_->SetUniform(Uniform("texDim", (float)TEXTURE_DIMENSION));
  update_prog_->SetUniform(Uniform("pose_inv", pose_inv));

  glBindBuffer(GL_ARRAY_BUFFER, vbos_[target_].first);

  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, 0);

  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE,
                        reinterpret_cast<GLvoid *>(sizeof(Eigen::Vector4f)));

  glEnableVertexAttribArray(2);
  glVertexAttribPointer(
      2, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE,
      reinterpret_cast<GLvoid *>(sizeof(Eigen::Vector4f) * 2));

  glEnable(GL_RASTERIZER_DISCARD);

  glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, vbos_[render_].second);

  glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, vbos_[render_].first);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, update_map_verts_confs_.texture->tid);

  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, update_map_colors_time_.texture->tid);

  glActiveTexture(GL_TEXTURE2);
  glBindTexture(GL_TEXTURE_2D, update_map_norms_radii_.texture->tid);

#ifdef DEBUG_UPDATE_PROG
  glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, count_query_);
#endif

  glBeginTransformFeedback(GL_POINTS);

  glDrawTransformFeedback(GL_POINTS, vbos_[target_].second);

#ifdef DEBUG_UPDATE_PROG
  glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);

  unsigned int update_count;
  glGetQueryObjectuiv(count_query_, GL_QUERY_RESULT, &update_count);

  LOG(INFO) << "udpate count: " << update_count;
#endif

  glEndTransformFeedback();

  glDisable(GL_RASTERIZER_DISCARD);

  glDisableVertexAttribArray(0);
  glDisableVertexAttribArray(1);
  glDisableVertexAttribArray(2);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);

  update_prog_->Unbind();

  std::swap(target_, render_);

  glFinish();
}

void GlobalModel::RenderPointCloud(pangolin::OpenGlMatrix mvp,
                                   const bool draw_normals,
                                   const bool draw_colors) {
  std::shared_ptr<Shader> program = draw_surfel_prog_;

  program->Bind();

  program->SetUniform(Uniform("MVP", mvp));
  program->SetUniform(
      Uniform("colorType", (draw_normals ? 1 : draw_colors ? 2 : 0)));

  Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
  // This is for the point shader
  program->SetUniform(Uniform("pose", pose));

  glBindBuffer(GL_ARRAY_BUFFER, vbos_[target_].first);

  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, 0);
  //  glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);

  // NOTE: for test
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(
      1, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE,
      reinterpret_cast<GLvoid *>(sizeof(Eigen::Vector4f) * 1));

  glEnableVertexAttribArray(2);
  glVertexAttribPointer(
      2, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE,
      reinterpret_cast<GLvoid *>(sizeof(Eigen::Vector4f) * 2));

  glDrawTransformFeedback(GL_POINTS, vbos_[target_].second);

  glDisableVertexAttribArray(0);
  // NOTE: for test
  glDisableVertexAttribArray(1);
  glDisableVertexAttribArray(2);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  program->Unbind();
}

Eigen::Vector4f *GlobalModel::DownloadMap() {
  glFinish();

  Eigen::Vector4f *vertices = new Eigen::Vector4f[count_ * 3];

  memset(&vertices[0], 0, count_ * Vertex::SIZE);

  GLuint downloadVbo;

  glGenBuffers(1, &downloadVbo);
  glBindBuffer(GL_ARRAY_BUFFER, downloadVbo);
  glBufferData(GL_ARRAY_BUFFER, buffer_size_, 0, GL_STREAM_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  glBindBuffer(GL_COPY_READ_BUFFER, vbos_[render_].first);
  glBindBuffer(GL_COPY_WRITE_BUFFER, downloadVbo);

  glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0,
                      count_ * Vertex::SIZE);
  glGetBufferSubData(GL_COPY_WRITE_BUFFER, 0, count_ * Vertex::SIZE, vertices);

  glBindBuffer(GL_COPY_READ_BUFFER, 0);
  glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
  glDeleteBuffers(1, &downloadVbo);

  glFinish();

  return vertices;
}

}  // namespace dsl