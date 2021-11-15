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
// Created by hyye on 12/28/19.
//

#include "map_render/pangolin_utils.h"

namespace {
void SaveFigure(pangolin::GlTexture *texture, std::string filename_wo_extension,
                bool normalize = false) {
  std::string filename;
  pangolin::TypedImage download_img, saved_img;
  texture->Download(download_img);
  LOG(INFO) << "download_img: " << download_img.pitch << " " << download_img.w
            << " " << download_img.h;
  LOG(INFO) << std::string(download_img.fmt);

  if (std::string(download_img.fmt) == "GRAY16LE") {
    auto pf = pangolin::PixelFormatFromString("GRAY16LE");
    saved_img.Reinitialise(download_img.w, download_img.h, pf);
    LOG(INFO) << "saved_img: " << saved_img.pitch << " " << saved_img.w << ""
              << saved_img.h;
    LOG(INFO) << std::string(saved_img.fmt);

    for (int y = 0; y < download_img.h; ++y) {
      for (int x = 0; x < download_img.w; ++x) {
        *(uint16_t *)&saved_img.ptr[y * download_img.w * 2 + x * 2] =
            (uint16_t)(*(float *)&download_img
                            .ptr[y * (download_img.w * 4) + (x * 4)] *
                       65535);
      }
    }
    filename = filename_wo_extension + ".png";
  } else if (std::string(download_img.fmt) == "RGBA128F") {
    auto pf = pangolin::PixelFormatFromString("RGB24");
    saved_img.Reinitialise(download_img.w, download_img.h, pf);
    LOG(INFO) << "saved_img: " << saved_img.pitch << " " << saved_img.w << ""
              << saved_img.h;
    LOG(INFO) << std::string(saved_img.fmt);

    for (int y = 0; y < download_img.h; ++y) {
      for (int x = 0; x < download_img.w * 3; x += 3) {
        int x1 = x, x2 = x + 1, x3 = x + 2;
        float value1 =
            *(float *)&download_img.ptr[y * (download_img.w * 4 * 4) +
                                        (x1 / 3 * 4 * 4 + x1 % 3 * 4)];
        float value2 =
            *(float *)&download_img.ptr[y * (download_img.w * 4 * 4) +
                                        (x2 / 3 * 4 * 4 + x2 % 3 * 4)];
        float value3 =
            *(float *)&download_img.ptr[y * (download_img.w * 4 * 4) +
                                        (x3 / 3 * 4 * 4 + x3 % 3 * 4)];
        if (normalize && (value1 || value2 || value3)) {
          value1 = (value1 + 10) / 20;
          value2 = (value2 + 10) / 20;
          value3 = (value3) / 5;
        }
        value1 = fabs(value1);
        value2 = fabs(value2);
        value3 = fabs(value3);
        saved_img.ptr[y * download_img.w * 3 * 1 + x1 * 1] =
            (unsigned char)(value1 * 255);
        saved_img.ptr[y * download_img.w * 3 * 1 + x2 * 1] =
            (unsigned char)(value2 * 255);
        saved_img.ptr[y * download_img.w * 3 * 1 + x3 * 1] =
            (unsigned char)(value3 * 255);
      }
    }
    filename = filename_wo_extension + ".png";
  } else if (std::string(download_img.fmt) == "GRAY32F") {
    auto pf = pangolin::PixelFormatFromString("RGB24");
    saved_img.Reinitialise(download_img.w, download_img.h, pf);
    LOG(INFO) << "saved_img: " << saved_img.pitch << " " << saved_img.w << " "
              << saved_img.h;
    LOG(INFO) << std::string(saved_img.fmt);

    for (int y = 0; y < download_img.h; ++y) {
      for (int x = 0; x < download_img.w * 3; x += 3) {
        int x1 = x, x2 = x + 1, x3 = x + 2;
        float depth =
            *(float *)&download_img.ptr[y * (download_img.w * 4) + (x / 3 * 4)];
        int depth_int = depth / 1000 * (256 * 256 * 256 - 1);
        int b = depth_int / 256 / 256;
        int g = (depth_int - b * 256 * 256) / 256;
        int r = (depth_int - b * 256 * 256 - g * 256) % 256;
        saved_img.ptr[y * download_img.w * 3 * 1 + x1 * 1] = (unsigned char)r;
        saved_img.ptr[y * download_img.w * 3 * 1 + x2 * 1] = (unsigned char)g;
        saved_img.ptr[y * download_img.w * 3 * 1 + x3 * 1] = (unsigned char)b;
      }
    }
    filename = filename_wo_extension + ".png";
  } else {
    LOG(ERROR) << "save " << std::string(download_img.fmt)
               << " not implemented";
    return;
  }

  pangolin::SaveImage(saved_img, filename);
}
}  // namespace

namespace dsl {

bool PangolinUtils::SetVertexNormal(pangolin::GlTexture *vertex_tex,
                                    pangolin::GlTexture *normal_tex) {
  height = width = 0;

  vertex_tex_ = vertex_tex;
  normal_tex_ = normal_tex;
  vertex_available_ = false;
  normal_available_ = false;
  if (vertex_tex_ != NULL) {
    vertex_tex->Download(vertex_img);
    DLOG(INFO) << "vertex_img.fmt.format:" << vertex_img.fmt.format;
    LOG_ASSERT(vertex_img.fmt.format == "RGBA128F");
    vertex_available_ = true;
  } else {
    return false;
  }
  if (normal_tex_ != NULL) {
    normal_tex->Download(normal_img);
    DLOG(INFO) << "normal_img.fmt.format:" << normal_img.fmt.format;
    LOG_ASSERT(normal_img.fmt.format == "RGBA128F");
    normal_available_ = true;
  }

  height = vertex_img.h;
  width = vertex_img.w;

  return true;
}

/// Get vertex and normal as pcl point
/// \param x x-axis in the image corrdinate
/// \param y y-axis in the image corrdinate
/// \param pcl_point
/// \return valid point or not
bool PangolinUtils::GetPoint(float x, float y,
                             pcl::PointXYZINormal &pcl_point) {
  if (int(x) < 0 || int(x) >= width || int(y) < 0 || int(y) >= height) {
    LOG(WARNING) << "invalid image coordinates (x, y) " << x << ", " << y;
  }

  bool valid = false;

  if (vertex_available_) {
    int y_start = int(y) * (vertex_img.w * 4 * 4);
    int x_start = int(x) * 4 * 4;
    float *point_ptr = (float *)&vertex_img.ptr[y_start + x_start];
    pcl_point.x = *(point_ptr);
    pcl_point.y = *(point_ptr + 1);
    pcl_point.z = *(point_ptr + 2);

    if (normal_available_) {
      float *normal_ptr = (float *)&normal_img.ptr[y_start + x_start];
      pcl_point.normal_x = *(normal_ptr);
      pcl_point.normal_y = *(normal_ptr + 1);
      pcl_point.normal_z = *(normal_ptr + 2);
    }

    if (!pcl::isFinite(pcl_point) ||
        (pcl_point.x == 0 && pcl_point.y == 0 && pcl_point.z == 0) ||
        !(pcl_isfinite(pcl_point.normal_x) &&
          pcl_isfinite(pcl_point.normal_y) &&
          pcl_isfinite(pcl_point.normal_z)) ||
        (pcl_point.normal_x == 0 && pcl_point.normal_y == 0 &&
         pcl_point.normal_z == 0)) {
      valid = false;
    } else {
      valid = true;
    }
  }

  return valid;
}

void PangolinUtils::SaveFigures(std::vector<pangolin::GlTexture *> textures,
                                std::vector<bool> flags, int current_frame,
                                std::string saving_path) {
  pangolin::TypedImage download_img, saved_img;

  for (int i = 0; i < textures.size(); ++i) {
    auto texture = textures[i];
    bool flag = flags[i];
    SaveFigure(texture,
               saving_path + "/frame_" + std::to_string(current_frame) + "_" +
                   std::to_string(i),
               flag);
  }
}

}  // namespace dsl