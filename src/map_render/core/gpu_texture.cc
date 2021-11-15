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

#include "map_render/core/gpu_texture.h"

namespace dsl {

const std::string GPUTexture::RGB = "RGB";
const std::string GPUTexture::DEPTH_RAW = "DEPTH";
const std::string GPUTexture::DEPTH_FILTERED = "DEPTH_FILTERED";
const std::string GPUTexture::DEPTH_METRIC = "DEPTH_METRIC";
const std::string GPUTexture::DEPTH_METRIC_FILTERED = "DEPTH_METRIC_FILTERED";
const std::string GPUTexture::DEPTH_NORM = "DEPTH_NORM";
const std::string GPUTexture::VERTEX_MAP = "VERTEX_MAP";
const std::string GPUTexture::REF_RGB = "REF_RGB";
const std::string GPUTexture::PREDICT_DEPTH = "PREDICT_DEPTH";

GPUTexture::GPUTexture(const int width, const int height,
                       const GLenum internalFormat, const GLenum format,
                       const GLenum dataType, const bool draw, const bool cuda)
    : texture(new pangolin::GlTexture(width, height, internalFormat, draw, 0,
                                      format, dataType)),
      draw(draw),
      width(width),
      height(height),
      internalFormat(internalFormat),
      format(format),
      dataType(dataType) {
  if (cuda) {
    cudaGraphicsGLRegisterImage(&cudaRes, texture->tid, GL_TEXTURE_2D,
                                cudaGraphicsRegisterFlagsReadOnly);
  } else {
    cudaRes = 0;
  }
}

GPUTexture::~GPUTexture() {
  if (texture) {
    delete texture;
  }

  if (cudaRes) {
    cudaGraphicsUnregisterResource(cudaRes);
  }
}

}  // namespace dsl