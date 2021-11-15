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

#include "map_render/shaders/feedback_buffer.h"

namespace dsl {

const std::string FeedbackBuffer::RAW = "RAW";
const std::string FeedbackBuffer::FILTERED = "FILTERED";
const std::string FeedbackBuffer::PREDICT = "PREDICT";

FeedbackBuffer::FeedbackBuffer(std::shared_ptr<Shader> program)
    : program(program),
      drawProgram(LoadProgramFromFile("draw_feedback.vert", "draw_feedback.frag")),
      bufferSize(wG[0] * hG[0] * Vertex::SIZE),
      count(0)
{
  float * vertices = new float[bufferSize];

  memset(&vertices[0], 0, bufferSize);

  glGenTransformFeedbacks(1, &fid);
  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, bufferSize, &vertices[0], GL_STREAM_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  delete [] vertices;

  std::vector<Eigen::Vector2f> uv;

  for(int i = 0; i < wG[0]; i++)
  {
    for(int j = 0; j < hG[0]; j++)
    {
      uv.push_back(Eigen::Vector2f(((float)i / (float)wG[0]) + 1.0 / (2 * (float)wG[0]),
                                   ((float)j / (float)hG[0]) + 1.0 / (2 * (float)hG[0])));
    }
  }

  glGenBuffers(1, &uvo);
  glBindBuffer(GL_ARRAY_BUFFER, uvo);
  glBufferData(GL_ARRAY_BUFFER, uv.size() * sizeof(Eigen::Vector2f), &uv[0], GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  program->Bind();

  int loc[3] =
      {
          glGetVaryingLocationNV(program->ProgramId(), "vPosition0"),
          glGetVaryingLocationNV(program->ProgramId(), "vColor0"),
          glGetVaryingLocationNV(program->ProgramId(), "vNormRad0"),
      };

  glTransformFeedbackVaryingsNV(program->ProgramId(), 3, loc, GL_INTERLEAVED_ATTRIBS);

  program->Unbind();

  glGenQueries(1, &countQuery);
}

FeedbackBuffer::~FeedbackBuffer()
{
  glDeleteTransformFeedbacks(1, &fid);
  glDeleteBuffers(1, &vbo);
  glDeleteBuffers(1, &uvo);
  glDeleteQueries(1, &countQuery);
}

void FeedbackBuffer::compute(pangolin::GlTexture * color,
                             pangolin::GlTexture * depth,
                             const int & time,
                             const float depthCutoff)
{
  program->Bind();

  Eigen::Vector4f cam(cxG[0],
                      cyG[0],
                      1.0f / fxG[0],
                      1.0f / fyG[0]);

  program->SetUniform(Uniform("cam", cam));
  program->SetUniform(Uniform("threshold", 0.0f));
  program->SetUniform(Uniform("cols", (float)wG[0]));
  program->SetUniform(Uniform("rows", (float)hG[0]));
  program->SetUniform(Uniform("time", time));
  program->SetUniform(Uniform("gSampler", 0));
  program->SetUniform(Uniform("cSampler", 1));
  program->SetUniform(Uniform("maxDepth", depthCutoff));

  glEnableVertexAttribArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, uvo);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);

  glEnable(GL_RASTERIZER_DISCARD);

  glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, fid);

  glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, vbo);

  glBeginTransformFeedback(GL_POINTS);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, depth->tid);

  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, color->tid);

  glDrawArrays(GL_POINTS, 0, wG[0] * hG[0]);

  glBindTexture(GL_TEXTURE_2D, 0);

  glActiveTexture(GL_TEXTURE0);

  glEndTransformFeedback();

  glDisable(GL_RASTERIZER_DISCARD);

  glDisableVertexAttribArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);

  program->Unbind();

  glFinish();
}

void FeedbackBuffer::render(pangolin::OpenGlMatrix mvp,
                            const Eigen::Matrix4f & pose,
                            const bool drawNormals,
                            const bool drawColors)
{
  drawProgram->Bind();

  drawProgram->SetUniform(Uniform("MVP", mvp));
  drawProgram->SetUniform(Uniform("pose", pose));
  drawProgram->SetUniform(Uniform("colorType", (drawNormals ? 1 : drawColors ? 2 : 0)));

  glBindBuffer(GL_ARRAY_BUFFER, vbo);

  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, 0);

  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 1));

  glEnableVertexAttribArray(2);
  glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 2));

  glDrawTransformFeedback(GL_POINTS, fid);

  glDisableVertexAttribArray(0);
  glDisableVertexAttribArray(1);
  glDisableVertexAttribArray(2);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  drawProgram->Unbind();
}

}