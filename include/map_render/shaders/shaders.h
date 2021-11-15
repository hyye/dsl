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

#ifndef DSL_SHADERS_H_
#define DSL_SHADERS_H_

#include <pangolin/gl/glsl.h>
#include <memory>
#include "parse.h"
#include "uniform.h"

namespace dsl {

class Shader : public pangolin::GlSlProgram {
 public:
  Shader() {}

  GLuint ProgramId() { return prog; }

  void SetUniform(const Uniform &v) {
    GLuint loc = glGetUniformLocation(prog, v.id.c_str());

    switch (v.t) {
      case Uniform::INT:
        glUniform1i(loc, v.i);
        break;
      case Uniform::FLOAT:
        glUniform1f(loc, v.f);
        break;
      case Uniform::VEC2:
        glUniform2f(loc, v.v2(0), v.v2(1));
        break;
      case Uniform::VEC3:
        glUniform3f(loc, v.v3(0), v.v3(1), v.v3(2));
        break;
      case Uniform::VEC4:
        glUniform4f(loc, v.v4(0), v.v4(1), v.v4(2), v.v4(3));
        break;
      case Uniform::MAT4:
        glUniformMatrix4fv(loc, 1, false, v.m4.data());
        break;
      default:
        assert(false && "Uniform type not implemented!");
        break;
    }
  }
};

static inline std::shared_ptr<Shader> LoadProgramGeomFromFile(
    const std::string &vertex_shader_file,
    const std::string &geometry_shader_file) {
  std::shared_ptr<Shader> program = std::make_shared<Shader>();

  program->AddShaderFromFile(
      pangolin::GlSlVertexShader,
      Parse::Get().ShaderDir() + "/" + vertex_shader_file, {},
      {Parse::Get().ShaderDir()});
  program->AddShaderFromFile(
      pangolin::GlSlGeometryShader,
      Parse::Get().ShaderDir() + "/" + geometry_shader_file, {},
      {Parse::Get().ShaderDir()});
  program->Link();

  return program;
}

static inline std::shared_ptr<Shader> LoadProgramFromFile(
    const std::string &vertex_shader_file) {
  std::shared_ptr<Shader> program = std::make_shared<Shader>();

  program->AddShaderFromFile(
      pangolin::GlSlVertexShader,
      Parse::Get().ShaderDir() + "/" + vertex_shader_file, {},
      {Parse::Get().ShaderDir()});
  program->Link();

  return program;
}

static inline std::shared_ptr<Shader> LoadProgramFromFile(
    const std::string &vertex_shader_file,
    const std::string &fragment_shader_file) {
  std::shared_ptr<Shader> program = std::make_shared<Shader>();

  program->AddShaderFromFile(
      pangolin::GlSlVertexShader,
      Parse::Get().ShaderDir() + "/" + vertex_shader_file, {},
      {Parse::Get().ShaderDir()});
  program->AddShaderFromFile(
      pangolin::GlSlFragmentShader,
      Parse::Get().ShaderDir() + "/" + fragment_shader_file, {},
      {Parse::Get().ShaderDir()});
  program->Link();

  return program;
}

static inline std::shared_ptr<Shader> LoadProgramFromFile(
    const std::string &vertex_shader_file,
    const std::string &fragment_shader_file,
    const std::string &geometry_shader_file) {
  std::shared_ptr<Shader> program = std::make_shared<Shader>();

  program->AddShaderFromFile(
      pangolin::GlSlVertexShader,
      Parse::Get().ShaderDir() + "/" + vertex_shader_file, {},
      {Parse::Get().ShaderDir()});
  program->AddShaderFromFile(
      pangolin::GlSlGeometryShader,
      Parse::Get().ShaderDir() + "/" + geometry_shader_file, {},
      {Parse::Get().ShaderDir()});
  program->AddShaderFromFile(
      pangolin::GlSlFragmentShader,
      Parse::Get().ShaderDir() + "/" + fragment_shader_file, {},
      {Parse::Get().ShaderDir()});
  program->Link();

  return program;
}

}  // namespace dsl

#endif  // DSL_SHADERS_H_
