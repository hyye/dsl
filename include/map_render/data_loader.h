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

#ifndef DSL_DATA_LOADER_H_
#define DSL_DATA_LOADER_H_

#include <pangolin/image/image_io.h>
#include <pangolin/gl/gl.h>
#include <Eigen/Core>
#include <boost/filesystem.hpp>
#include <string>
// #include "Img.h"
#include "csv.h"
#include "pcl_utils.h"
#include "shaders/vertex.h"
#include "util/global_calib.h"

namespace dsl {

class DataLoader {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  DataLoader(std::string file)
      : file_(file),
        width_(wG[0]),
        height_(hG[0]),
        num_pixels_(width_ * height_),
        current_frame(0) {}

  virtual ~DataLoader() {}
  virtual void GetNext(bool fast_forward = false) = 0;
  virtual int GetNumFrames() = 0;
  virtual void FastForward(int frame) = 0;
  virtual const std::string GetFile() = 0;
  virtual bool HasMore() = 0;

  //  std::unique_ptr<pangolin::TypedImage> input_image_ptr;
  pangolin::TypedImage input_image;

  Eigen::Affine3f pose;
  int current_frame;
  double timestamp;
  //  Img<Eigen::Matrix<unsigned char, 3, 1>> *input_image;

 protected:
  std::string file_;

  int width_;
  int height_;
  int num_pixels_;

  int num_frames_;
};

class CsvDataLoader : public DataLoader {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  CsvDataLoader(std::string csv_file, std::string pcd_file,
                double sample_ratio_ = 1, double intensity_scalar = 1.0,
                double rgb_scalar = 1.0, double surfel_size = 0.01,
                bool load_rgb = false);

  ~CsvDataLoader();

  void GetNext(bool fast_forward = false);

  int GetNumFrames();

  void FastForward(int frame);

  void ResetToBegin();

  const std::string GetFile();

  bool HasMore();

  void LoadImage(std::string image_name);

  pangolin::GlBuffer *GetVbo();

  boost::filesystem::path GetCsvFilePath() { return csv_file_path_; }

  int vertex_size;
  GLfloat *vertices;
  GLuint ptc_size = 0;
  PclUtils pcl_utils;

  std::string img_filename;

 protected:
  boost::filesystem::path csv_file_path_;
  std::string pcd_file_;
  std::unique_ptr<io::CSVReader<9> > csv_reader_;
  bool has_more_;
  double sample_ratio_;
  double rgb_scalar_ = 1.0;
  double surfel_size_;
};

class MapLoader {
 public:
  MapLoader(std::string pcd_file, double sample_ratio_ = 1,
            double intensity_scalar = 1.0, double rgb_scalar = 1.0,
            double surfel_size = 0.01, bool load_rgb = false);
  pangolin::GlBuffer *GetVbo();
  ~MapLoader();

  int vertex_size;
  GLfloat *vertices;
  GLuint ptc_size = 0;
  PclUtils pcl_utils;
};

}  // namespace dsl

#endif  // DSL_DATA_LOADER_H_
