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

#ifndef DSL_GPU_TEXTURE_H_
#define DSL_GPU_TEXTURE_H_

#include <pangolin/pangolin.h>
#include <driver_types.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>

namespace dsl {

class GPUTexture {
 public:
  GPUTexture(const int width, const int height, const GLenum internalFormat,
             const GLenum format, const GLenum dataType,
             const bool draw = false, const bool cuda = false);

  virtual ~GPUTexture();

  static const std::string RGB, DEPTH_RAW, DEPTH_FILTERED, DEPTH_METRIC,
      DEPTH_METRIC_FILTERED, DEPTH_NORM, VERTEX_MAP, REF_RGB, PREDICT_DEPTH;

  pangolin::GlTexture *texture;

  cudaGraphicsResource *cudaRes;

  const bool draw;

 private:
  GPUTexture()
      : texture(0),
        cudaRes(0),
        draw(false),
        width(0),
        height(0),
        internalFormat(0),
        format(0),
        dataType(0) {}
  const int width;
  const int height;
  const GLenum internalFormat;
  const GLenum format;
  const GLenum dataType;
};

}

#endif  // DSL_GPU_TEXTURE_H_
