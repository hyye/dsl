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

#include "map_render/core/index_map.h"

namespace dsl {

IndexMap::IndexMap()
    : index_prog_(LoadProgramFromFile("index_map_cata.vert", "index_map.frag")),
      index_renderbuffer_(wG[0], hG[0]),
      index_texture_(wG[0], hG[0],
                     GL_R32UI,        // GL_LUMINANCE32UI_EXT,
                     GL_RED_INTEGER,  // GL_LUMINANCE_INTEGER_EXT,
                     GL_UNSIGNED_INT),
      vert_conf_texture_(wG[0], hG[0], GL_RGBA32F, GL_RGB /*GL_LUMINANCE*/,
                         GL_FLOAT),
      color_time_texture_(wG[0], hG[0], GL_RGBA32F, GL_RGB /*GL_LUMINANCE*/,
                          GL_FLOAT),
      normal_rad_texture_(wG[0], hG[0], GL_RGBA32F, GL_RGB /*GL_LUMINANCE*/,
                          GL_FLOAT),

      index_global_prog_(LoadProgramFromFile("index_global_map.vert",
                                             "index_global_map.frag",
                                             "index_global_map_cata.geom")),
      index_global_renderbuffer_(wG[0], hG[0]),
      index_global_texture_(wG[0], hG[0],
                            GL_R32UI,        // GL_LUMINANCE32UI_EXT,
                            GL_RED_INTEGER,  // GL_LUMINANCE_INTEGER_EXT,
                            GL_UNSIGNED_INT),
      vert_global_texture_(wG[0], hG[0], GL_RGBA32F, GL_RGB /*GL_LUMINANCE*/,
                           GL_FLOAT),
      color_global_texture_(wG[0], hG[0], GL_RGBA32F, GL_RGB /*GL_LUMINANCE*/,
                            GL_FLOAT),
      normal_global_texture_(wG[0], hG[0], GL_RGBA32F, GL_RGB /*GL_LUMINANCE*/,
                             GL_FLOAT),

      draw_depth_prog_(LoadProgramFromFile(
          "empty.vert", "visualise_textures.frag", "quad.geom")),
      draw_renderbuffer_(wG[0], hG[0]),
      draw_texture_(wG[0], hG[0],
                    GL_RGBA,  // use GL_RGB8 to enable saving
                    GL_RGB, GL_UNSIGNED_BYTE, false),
      intensity_prog_(LoadProgramFromFile("splat_cata.vert",
                                          "intensity_splat_cata.frag")),
      intensity_renderbuffer_(wG[0], hG[0]),
      intensity_texture_(wG[0], hG[0],
                         GL_RGBA,  // use GL_RGB8 to enable saving
                         GL_RGB, GL_UNSIGNED_BYTE, false),
      depth_prog_(LoadProgramFromFile("splat_cata.vert",
                                      "depth_splat_cata.frag")),
      depth_renderbuffer_(wG[0], hG[0]),
      depth_texture_(wG[0], hG[0],
                     GL_LUMINANCE32F_ARB,  // GL_R32F
                     GL_LUMINANCE,         // GL_RED
                     GL_FLOAT, false, true),
      combined_prog_(LoadProgramFromFile("splat_cata.vert",
                                         "combo_splat_cata.frag")),
      combined_renderbuffer_(wG[0], hG[0]),
      image_texture_(wG[0], hG[0],
                     GL_RGBA,  // use GL_RGB8 to enable saving
                     GL_RGB, GL_UNSIGNED_BYTE, false, true),
      vertex_texture_(wG[0], hG[0], GL_RGBA32F, GL_LUMINANCE, GL_FLOAT, false,
                      true),
      normal_texture_(wG[0], hG[0], GL_RGBA32F, GL_LUMINANCE, GL_FLOAT, false,
                      true) {
  index_framebuffer_.AttachColour(*index_texture_.texture);
  index_framebuffer_.AttachColour(*vert_conf_texture_.texture);
  index_framebuffer_.AttachColour(*color_time_texture_.texture);
  index_framebuffer_.AttachColour(*normal_rad_texture_.texture);
  index_framebuffer_.AttachDepth(index_renderbuffer_);

  index_global_framebuffer_.AttachColour(*index_global_texture_.texture);
  index_global_framebuffer_.AttachColour(*vert_global_texture_.texture);
  index_global_framebuffer_.AttachColour(*color_global_texture_.texture);
  index_global_framebuffer_.AttachColour(*normal_global_texture_.texture);
  index_global_framebuffer_.AttachDepth(index_global_renderbuffer_);

  draw_framebuffer_.AttachColour(*draw_texture_.texture);
  draw_framebuffer_.AttachDepth(draw_renderbuffer_);

  intensity_framebuffer_.AttachColour(*intensity_texture_.texture);
  intensity_framebuffer_.AttachDepth(intensity_renderbuffer_);

  depth_framebuffer_.AttachColour(*depth_texture_.texture);
  depth_framebuffer_.AttachDepth(depth_renderbuffer_);

  combined_framebuffer_.AttachColour(*image_texture_.texture);
  combined_framebuffer_.AttachColour(*vertex_texture_.texture);
  combined_framebuffer_.AttachColour(*normal_texture_.texture);
  combined_framebuffer_.AttachDepth(combined_renderbuffer_);

  /*  { ///
      glGenBuffers(1, &tbo_);
      glBindBuffer(GL_ARRAY_BUFFER, tbo_);
      glBufferData(GL_ARRAY_BUFFER, 1000 * 1000 * sizeof(float) * 4, nullptr,
    GL_STREAM_DRAW); glBindBuffer(GL_ARRAY_BUFFER, 0);

      depth_prog_->Bind();

      int locInit[1] =
          {
              glGetVaryingLocationNV(depth_prog_->programId(), "position")
          };

      glTransformFeedbackVaryingsNV(depth_prog_->programId(), 1, locInit,
    GL_INTERLEAVED_ATTRIBS);

      glGenQueries(1, &count_query_);

      depth_prog_->Unbind();
    } ///*/

#ifdef DEBUG_INDEXMAP
  glGenBuffers(1, &tbo_);
  glBindBuffer(GL_ARRAY_BUFFER, tbo_);
  glBufferData(GL_ARRAY_BUFFER, 3072 * 3072 * sizeof(GLuint) * 5, nullptr,
               GL_STREAM_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  index_prog_->Bind();
  int locInit[2] = {
      glGetVaryingLocationNV(index_prog_->programId(), "vertexId"),
      glGetVaryingLocationNV(index_prog_->programId(), "vPosition0")};

  glTransformFeedbackVaryingsNV(index_prog_->programId(), 2, locInit,
                                GL_INTERLEAVED_ATTRIBS);

  glGenQueries(1, &count_query_);
  index_prog_->Unbind();
#endif
}

IndexMap::~IndexMap() {}

void IndexMap::SynthesizeDepth(const Eigen::Matrix4f &pose,
                               const std::pair<GLuint, GLuint> &model,
                               const float depth_cutoff) {
  glEnable(GL_PROGRAM_POINT_SIZE);
  glEnable(GL_POINT_SPRITE);

  depth_framebuffer_.Bind();

  glPushAttrib(GL_VIEWPORT_BIT);

  glViewport(0, 0, depth_renderbuffer_.width, depth_renderbuffer_.height);

  glClearColor(0, 0, 0, 1);

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  depth_prog_->Bind();

  Eigen::Matrix4f t_inv = pose.inverse();

  //  Eigen::Vector4f cam(Intrinsics::getInstance().cx(),
  //                      Intrinsics::getInstance().cy(),
  //                      Intrinsics::getInstance().fx(),
  //                      Intrinsics::getInstance().fy());
  Eigen::Vector4f cam(cxG[0], cyG[0], fxG[0], fyG[0]);
  float xi = xiG;
  float max_half_fov = max_half_fovG;

  depth_prog_->SetUniform(Uniform("t_inv", t_inv));
  depth_prog_->SetUniform(Uniform("cam", cam));
  depth_prog_->SetUniform(Uniform("xi", xi));
  depth_prog_->SetUniform(Uniform("max_half_fov", max_half_fov));
  depth_prog_->SetUniform(Uniform("maxDepth", depth_cutoff));
  //  depth_prog_->SetUniform(Uniform("confThreshold", confThreshold));
  depth_prog_->SetUniform(Uniform("cols", (float)wG[0]));
  depth_prog_->SetUniform(Uniform("rows", (float)hG[0]));
  //  depth_prog_->SetUniform(Uniform("time", time));
  //  depth_prog_->SetUniform(Uniform("maxTime", maxTime));
  //  depth_prog_->SetUniform(Uniform("timeDelta", timeDelta));

  glBindBuffer(GL_ARRAY_BUFFER, model.first);

  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, 0);
  //  glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);

  glEnableVertexAttribArray(1);
  glVertexAttribPointer(
      1, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE,
      reinterpret_cast<GLvoid *>(sizeof(Eigen::Vector4f) * 1));

  glEnableVertexAttribArray(2);
  glVertexAttribPointer(
      2, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE,
      reinterpret_cast<GLvoid *>(sizeof(Eigen::Vector4f) * 2));

  /*  ///
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, tbo_);

    glBeginTransformFeedback(GL_POINTS);

    glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, count_query_);
    ///*/

  glDrawTransformFeedback(GL_POINTS, model.second);

  /*  ///
    glEndTransformFeedback();

    glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);

    glGetQueryObjectuiv(count_query_, GL_QUERY_RESULT, &count_);

    LOG(INFO) << "count_: " << count_;
    ///*/

  glDisableVertexAttribArray(0);
  glDisableVertexAttribArray(1);
  glDisableVertexAttribArray(2);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  depth_framebuffer_.Unbind();

  depth_prog_->Unbind();

  glDisable(GL_PROGRAM_POINT_SIZE);
  glDisable(GL_POINT_SPRITE);

  glPopAttrib();

  glFinish();
}

void IndexMap::PredictIndices(const Eigen::Matrix4f &pose,
                              const std::pair<GLuint, GLuint> &model,
                              const float depth_cutoff) {
  index_framebuffer_.Bind();

  glPushAttrib(GL_VIEWPORT_BIT);

  glViewport(0, 0, index_renderbuffer_.width, index_renderbuffer_.height);

  glClearColor(0, 0, 0, 0);

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  index_prog_->Bind();

  Eigen::Matrix4f t_inv = pose.inverse();

  //  Eigen::Vector4f cam(Intrinsics::getInstance().cx(),
  //                      Intrinsics::getInstance().cy(),
  //                      Intrinsics::getInstance().fx(),
  //                      Intrinsics::getInstance().fy());
  Eigen::Vector4f cam(cxG[0], cyG[0], fxG[0], fyG[0]);
  float xi = xiG;
  float max_half_fov = max_half_fovG;

  index_prog_->SetUniform(Uniform("t_inv", t_inv));
  index_prog_->SetUniform(Uniform("cam", cam));
  index_prog_->SetUniform(Uniform("xi", xi));
  index_prog_->SetUniform(Uniform("max_half_fov", max_half_fov));
  index_prog_->SetUniform(Uniform("maxDepth", depth_cutoff));
  index_prog_->SetUniform(Uniform("cols", (float)wG[0]));
  index_prog_->SetUniform(Uniform("rows", (float)hG[0]));

  glBindBuffer(GL_ARRAY_BUFFER, model.first);

  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, 0);

  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE,
                        reinterpret_cast<GLvoid *>(sizeof(Eigen::Vector4f)));

  glEnableVertexAttribArray(2);
  glVertexAttribPointer(
      2, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE,
      reinterpret_cast<GLvoid *>(sizeof(Eigen::Vector4f) * 2));

#ifdef DEBUG_INDEXMAP
  ///
  glEnable(GL_RASTERIZER_DISCARD);

  glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, tbo_);

  glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, count_query_);

  glBeginTransformFeedback(GL_POINTS);
  ///
#endif

  glDrawTransformFeedback(GL_POINTS, model.second);

#ifdef DEBUG_INDEXMAP
  ///
  glEndTransformFeedback();

  glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);

  glGetQueryObjectuiv(count_query_, GL_QUERY_RESULT, &count_);

  LOG(INFO) << "count_ in PredictIndices: " << count_;

  glDisable(GL_RASTERIZER_DISCARD);

  GLuint *feedback;
  feedback = new GLuint[count_ * 5];
  LOG(INFO) << "sizeof(feedback): " << sizeof(feedback);
  glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0,
                     count_ * 5 * sizeof(GLuint), feedback);

  for (int i = 0; i < 30; i++) {
    printf("%u, %f, %f, %f, %f\n", feedback[i * 5],
           *((float *)&feedback[i * 5 + 1]), *((float *)&feedback[i * 5 + 2]),
           *((float *)&feedback[i * 5 + 3]), *((float *)&feedback[i * 5 + 4]));
  }
  LOG(INFO) << std::endl;
  ///
#endif

  glDisableVertexAttribArray(0);
  glDisableVertexAttribArray(1);
  glDisableVertexAttribArray(2);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  index_framebuffer_.Unbind();

  index_prog_->Unbind();

  glPopAttrib();

  glFinish();
}

// repeated
void IndexMap::PredictGlobalIndices(const Eigen::Matrix4f &pose,
                                    const std::pair<GLuint, GLuint> &model,
                                    const float depth_cutoff) {
  index_global_framebuffer_.Bind();

  glPushAttrib(GL_VIEWPORT_BIT);

  glViewport(0, 0, index_global_renderbuffer_.width,
             index_global_renderbuffer_.height);

  glClearColor(0, 0, 0, 0);

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  index_global_prog_->Bind();

  Eigen::Matrix4f t_inv = pose.inverse();

  //  Eigen::Vector4f cam(Intrinsics::getInstance().cx(),
  //                      Intrinsics::getInstance().cy(),
  //                      Intrinsics::getInstance().fx(),
  //                      Intrinsics::getInstance().fy());
  Eigen::Vector4f cam(cxG[0], cyG[0], fxG[0], fyG[0]);
  float xi = xiG;
  float max_half_fov = max_half_fovG;

  index_global_prog_->SetUniform(Uniform("t_inv", t_inv));
  index_global_prog_->SetUniform(Uniform("cam", cam));
  index_global_prog_->SetUniform(Uniform("xi", xi));
  index_global_prog_->SetUniform(Uniform("max_half_fov", max_half_fov));
  index_global_prog_->SetUniform(Uniform("maxDepth", depth_cutoff));
  index_global_prog_->SetUniform(Uniform("cols", (float)wG[0]));
  index_global_prog_->SetUniform(Uniform("rows", (float)hG[0]));

  glBindBuffer(GL_ARRAY_BUFFER, model.first);

  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, 0);

  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE,
                        reinterpret_cast<GLvoid *>(sizeof(Eigen::Vector4f)));

  glEnableVertexAttribArray(2);
  glVertexAttribPointer(
      2, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE,
      reinterpret_cast<GLvoid *>(sizeof(Eigen::Vector4f) * 2));

#ifdef DEBUG_INDEXMAP
  ///
  glEnable(GL_RASTERIZER_DISCARD);

  glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, tbo_);

  glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, count_query_);

  glBeginTransformFeedback(GL_POINTS);
  ///
#endif

  glDrawTransformFeedback(GL_POINTS, model.second);

#ifdef DEBUG_INDEXMAP
  ///
  glEndTransformFeedback();

  glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);

  glGetQueryObjectuiv(count_query_, GL_QUERY_RESULT, &count_);

  LOG(INFO) << "count_ in PredictIndices: " << count_;

  glDisable(GL_RASTERIZER_DISCARD);

  GLuint *feedback;
  feedback = new GLuint[count_ * 5];
  LOG(INFO) << "sizeof(feedback): " << sizeof(feedback);
  glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0,
                     count_ * 5 * sizeof(GLuint), feedback);

  for (int i = 0; i < 30; i++) {
    printf("%u, %f, %f, %f, %f\n", feedback[i * 5],
           *((float *)&feedback[i * 5 + 1]), *((float *)&feedback[i * 5 + 2]),
           *((float *)&feedback[i * 5 + 3]), *((float *)&feedback[i * 5 + 4]));
  }
  LOG(INFO) << std::endl;
  ///
#endif

  glDisableVertexAttribArray(0);
  glDisableVertexAttribArray(1);
  glDisableVertexAttribArray(2);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  index_global_framebuffer_.Unbind();

  index_global_prog_->Unbind();

  glPopAttrib();

  glFinish();
}

void IndexMap::SynthesizeIntensity(const Eigen::Matrix4f &pose,
                                   const std::pair<GLuint, GLuint> &model,
                                   const float depth_cutoff) {
  glEnable(GL_PROGRAM_POINT_SIZE);
  glEnable(GL_POINT_SPRITE);

  intensity_framebuffer_.Bind();

  glPushAttrib(GL_VIEWPORT_BIT);

  glViewport(0, 0, depth_renderbuffer_.width, depth_renderbuffer_.height);

  glClearColor(0, 0, 0, 1);

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  intensity_prog_->Bind();

  Eigen::Matrix4f t_inv = pose.inverse();

  //  Eigen::Vector4f cam(Intrinsics::getInstance().cx(),
  //                      Intrinsics::getInstance().cy(),
  //                      Intrinsics::getInstance().fx(),
  //                      Intrinsics::getInstance().fy());
  Eigen::Vector4f cam(cxG[0], cyG[0], fxG[0], fyG[0]);
  float xi = xiG;
  float max_half_fov = max_half_fovG;

  intensity_prog_->SetUniform(Uniform("t_inv", t_inv));
  intensity_prog_->SetUniform(Uniform("cam", cam));
  intensity_prog_->SetUniform(Uniform("xi", xi));
  intensity_prog_->SetUniform(Uniform("max_half_fov", max_half_fov));
  intensity_prog_->SetUniform(Uniform("maxDepth", depth_cutoff));
  intensity_prog_->SetUniform(Uniform("cols", (float)wG[0]));
  intensity_prog_->SetUniform(Uniform("rows", (float)hG[0]));

  glBindBuffer(GL_ARRAY_BUFFER, model.first);

  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, 0);

  glEnableVertexAttribArray(1);
  glVertexAttribPointer(
      1, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE,
      reinterpret_cast<GLvoid *>(sizeof(Eigen::Vector4f) * 1));

  glEnableVertexAttribArray(2);
  glVertexAttribPointer(
      2, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE,
      reinterpret_cast<GLvoid *>(sizeof(Eigen::Vector4f) * 2));

  glDrawTransformFeedback(GL_POINTS, model.second);

  glDisableVertexAttribArray(0);
  glDisableVertexAttribArray(1);
  glDisableVertexAttribArray(2);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  intensity_framebuffer_.Unbind();

  intensity_prog_->Unbind();

  glDisable(GL_PROGRAM_POINT_SIZE);
  glDisable(GL_POINT_SPRITE);

  glPopAttrib();

  glFinish();
}

void IndexMap::RenderDepth(const float depth_cutoff) {
  draw_framebuffer_.Bind();

  glPushAttrib(GL_VIEWPORT_BIT);

  glViewport(0, 0, draw_renderbuffer_.width, draw_renderbuffer_.height);

  glClearColor(0, 0, 0, 1);

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  draw_depth_prog_->Bind();

  draw_depth_prog_->SetUniform(Uniform("maxDepth", depth_cutoff));

  glActiveTexture(GL_TEXTURE0);
  //  glBindTexture(GL_TEXTURE_2D, vertex_texture_.texture->tid);
  glBindTexture(GL_TEXTURE_2D, depth_texture_.texture->tid);

  draw_depth_prog_->SetUniform(Uniform("texVerts", 0));

  glDrawArrays(GL_POINTS, 0, 1);

  draw_framebuffer_.Unbind();

  draw_depth_prog_->Unbind();

  glBindTexture(GL_TEXTURE_2D, 0);

  glPopAttrib();

  glFinish();
}

void IndexMap::RenderDepth(const float depth_cutoff,
                           pangolin::GlTexture *vertex_texture) {
  draw_framebuffer_.Bind();

  glPushAttrib(GL_VIEWPORT_BIT);

  glViewport(0, 0, draw_renderbuffer_.width, draw_renderbuffer_.height);

  glClearColor(0, 0, 0, 1);

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  draw_depth_prog_->Bind();

  draw_depth_prog_->SetUniform(Uniform("maxDepth", depth_cutoff));

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, vertex_texture->tid);

  draw_depth_prog_->SetUniform(Uniform("texVerts", 0));

  glDrawArrays(GL_POINTS, 0, 1);

  draw_framebuffer_.Unbind();

  draw_depth_prog_->Unbind();

  glBindTexture(GL_TEXTURE_2D, 0);

  glPopAttrib();

  glFinish();
}

void IndexMap::CombinedPredict(const Eigen::Matrix4f &pose,
                               const std::pair<GLuint, GLuint> &model,
                               const float depth_cutoff) {
  glEnable(GL_PROGRAM_POINT_SIZE);
  glEnable(GL_POINT_SPRITE);

  combined_framebuffer_.Bind();

  glPushAttrib(GL_VIEWPORT_BIT);
  glViewport(0, 0, combined_renderbuffer_.width, combined_renderbuffer_.height);

  glClearColor(0, 0, 0, 0);

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  Eigen::Matrix4f t_inv = pose.inverse();

  //  Eigen::Vector4f cam(Intrinsics::getInstance().cx(),
  //                      Intrinsics::getInstance().cy(),
  //                      Intrinsics::getInstance().fx(),
  //                      Intrinsics::getInstance().fy());

  Eigen::Vector4f cam(cxG[0], cyG[0], fxG[0], fyG[0]);
  float xi = xiG;
  float max_half_fov = max_half_fovG;

  //  LOG(INFO) << cam.transpose() << " " << xi;

  combined_prog_->Bind();

  combined_prog_->SetUniform(Uniform("t_inv", t_inv));
  combined_prog_->SetUniform(Uniform("cam", cam));
  combined_prog_->SetUniform(Uniform("xi", xi));
  combined_prog_->SetUniform(Uniform("max_half_fov", max_half_fov));
  combined_prog_->SetUniform(Uniform("maxDepth", depth_cutoff));
  combined_prog_->SetUniform(Uniform("cols", (float)wG[0]));
  combined_prog_->SetUniform(Uniform("rows", (float)hG[0]));

  glBindBuffer(GL_ARRAY_BUFFER, model.first);

  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, 0);

  glEnableVertexAttribArray(1);
  glVertexAttribPointer(
      1, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE,
      reinterpret_cast<GLvoid *>(sizeof(Eigen::Vector4f) * 1));

  glEnableVertexAttribArray(2);
  glVertexAttribPointer(
      2, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE,
      reinterpret_cast<GLvoid *>(sizeof(Eigen::Vector4f) * 2));

  glDrawTransformFeedback(GL_POINTS, model.second);

  glDisableVertexAttribArray(0);
  glDisableVertexAttribArray(1);
  glDisableVertexAttribArray(2);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  combined_framebuffer_.Unbind();

  combined_prog_->Unbind();

  glDisable(GL_PROGRAM_POINT_SIZE);
  glDisable(GL_POINT_SPRITE);

  glPopAttrib();

  glFinish();
}

}  // namespace dsl