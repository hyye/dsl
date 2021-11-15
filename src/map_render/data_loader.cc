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

#include "map_render/data_loader.h"

namespace dsl {

CsvDataLoader::CsvDataLoader(std::string csv_file, std::string pcd_file,
                             double sample_ratio, double intensity_scalar,
                             double rgb_scalar, double surfel_size,
                             bool load_rgb)
    : DataLoader(csv_file),
      csv_file_path_(csv_file),
      pcd_file_(pcd_file),
      has_more_(false),
      vertex_size(Vertex::SIZE / sizeof(float)),
      sample_ratio_(sample_ratio),
      rgb_scalar_(rgb_scalar),
      surfel_size_(surfel_size) {
  {
    // preload
    csv_reader_ = std::make_unique<io::CSVReader<9> >(csv_file);
    csv_reader_->read_header(io::ignore_extra_column, "t", "px", "py", "pz",
                             "ow", "ox", "oy", "oz", "filename");
    double t, px, py, pz, ow, ox, oy, oz;
    num_frames_ = -1;
    std::string img_filename;
    while (csv_reader_->read_row(t, px, py, pz, ow, ox, oy, oz, img_filename)) {
      ++num_frames_;
    }
  }

  csv_reader_ = std::make_unique<io::CSVReader<9> >(csv_file);
  csv_reader_->read_header(io::ignore_extra_column, "t", "px", "py", "pz", "ow",
                           "ox", "oy", "oz", "filename");

  int vertex_size = Vertex::SIZE / sizeof(float);
  if (pcd_file_ != "") {
    pcl_utils.LoadPcd(pcd_file_, sample_ratio_, false);

    if (!load_rgb) {
      PointCloud ptc_tmp = *(pcl_utils.GetPointCloud());
      //    LOG(INFO) << "point cloud size: " << ptc_tmp.size();

      /*
      pcl_utils.SavePcd(
          "/mnt/HDD/Datasets/Visual/HKUST/li_cam/data_190107/validate_wide/predict_odom/flying_field_wide_predict_1901071222_with_normal.pcd");
      */

      PointCloud ptc;

      std::vector<int> indices;
      pcl::removeNaNFromPointCloud(ptc_tmp, ptc, indices);
      ptc_size = ptc.size();
      LOG(INFO) << "point cloud size: " << ptc_size;

      vertices = new float[ptc_size * vertex_size];

      for (int i = 0; i < ptc_size; ++i) {
        //      int rgb = int(round(ptc[i].curvature * 255.0f));
        //      rgb = (rgb << 8) + int(round(ptc[i].curvature * 255.0f));
        //      rgb = (rgb << 8) + int(round(ptc[i].curvature * 255.0f));

        vertices[i * vertex_size] = ptc[i].x;
        vertices[i * vertex_size + 1] = ptc[i].y;
        vertices[i * vertex_size + 2] = ptc[i].z;
        vertices[i * vertex_size + 3] = 0;
        vertices[i * vertex_size + 4] = 0;
        vertices[i * vertex_size + 5] = -1;  // -1 color undefined, but will be
                                             // reset in init_test_unstable.vert
        vertices[i * vertex_size + 6] = 0;
        // WARNING: temporal use for intensity
        vertices[i * vertex_size + 7] =
            int(ptc[i].intensity * intensity_scalar);
        vertices[i * vertex_size + 8] = ptc[i].normal[0];
        vertices[i * vertex_size + 9] = ptc[i].normal[1];
        vertices[i * vertex_size + 10] = ptc[i].normal[2];
        vertices[i * vertex_size + 11] = surfel_size_;

        //      std::cout << ptc[i].curvature << std::endl;
      }
    } else {
      RGBPointCloud ptc_tmp = *(pcl_utils.GetRGBPointCloud());
      RGBPointCloud ptc;

      std::vector<int> indices;
      pcl::removeNaNFromPointCloud(ptc_tmp, ptc, indices);
      ptc_size = ptc.size();
      LOG(INFO) << "point cloud size: " << ptc_size;

      vertices = new float[ptc_size * vertex_size];

      for (int i = 0; i < ptc_size; ++i) {
        int rgb = int(ptc[i].r);
        rgb = (rgb << 8) + int(ptc[i].g);
        rgb = (rgb << 8) + int(ptc[i].b);
        int gray = 0.299 * ptc[i].r + 0.587 * ptc[i].g + 0.114 * ptc[i].b;

        vertices[i * vertex_size] = ptc[i].x;
        vertices[i * vertex_size + 1] = ptc[i].y;
        vertices[i * vertex_size + 2] = ptc[i].z;
        vertices[i * vertex_size + 3] = 0;
        vertices[i * vertex_size + 4] = rgb;
        vertices[i * vertex_size + 5] = 1;  // -1 color undefined, but will be
                                            // reset in init_test_unstable.vert
        vertices[i * vertex_size + 6] = 0;
        // WARNING: temporal use for intensity
        vertices[i * vertex_size + 7] = gray;
        vertices[i * vertex_size + 8] = ptc[i].normal[0];
        vertices[i * vertex_size + 9] = ptc[i].normal[1];
        vertices[i * vertex_size + 10] = ptc[i].normal[2];
        vertices[i * vertex_size + 11] = surfel_size_;
      }
    }
  }

  has_more_ = true;

  // initially point to 0
  current_frame = -1;
  /*  GetNext();

    LOG_IF(FATAL, current_frame != 0) << "current_frame should equal to 0";*/
}

CsvDataLoader::~CsvDataLoader() { delete[] vertices; }

void CsvDataLoader::LoadImage(std::string image_name) {
  input_image = pangolin::LoadImage(img_filename);

  if (rgb_scalar_ != 1.0) {
    for (int i = 0; i < input_image.h * input_image.pitch; ++i) {
      int scaled_color = input_image[i] * rgb_scalar_;
      input_image[i] = (scaled_color > 255 ? 255 : scaled_color);
    }
  }
}

void CsvDataLoader::GetNext(bool fast_forward) {
  double t, px, py, pz, ow, ox, oy, oz;
  if (!csv_reader_->read_row(t, px, py, pz, ow, ox, oy, oz, img_filename)) {
    LOG(INFO) << "End of file.";
    LOG_ASSERT(num_frames_ == current_frame)
        << " " << num_frames_ << " " << current_frame;
    has_more_ = false;
    return;
  }

  boost::filesystem::path path_img(img_filename);

  if (path_img.is_relative()) {
    img_filename =
        (csv_file_path_.parent_path() / boost::filesystem::path(img_filename))
            .string();
  }

  if (!fast_forward) {
    LoadImage(img_filename);
  }

  timestamp = t;

  //  LOG(INFO) << std::setprecision(15) << t << " " << img_filename;

  pose.translation() << px, py, pz;
  pose.linear() = (Eigen::Quaternionf(ow, ox, oy, oz).toRotationMatrix());

  ++current_frame;
}

void CsvDataLoader::ResetToBegin() {
  csv_reader_ = std::make_unique<io::CSVReader<9> >(file_);
  csv_reader_->read_header(io::ignore_extra_column, "t", "px", "py", "pz", "ow",
                           "ox", "oy", "oz", "filename");

  has_more_ = true;
  // initially point to 0
  current_frame = -1;
}

void CsvDataLoader::FastForward(int frame) {
  if (current_frame >= frame) {
    ResetToBegin();
  }

  do {
    GetNext(true);
  } while (current_frame < frame);
  LoadImage(img_filename);
}

const std::string CsvDataLoader::GetFile() { return file_; }

int CsvDataLoader::GetNumFrames() { return num_frames_; }

bool CsvDataLoader::HasMore() { return has_more_; }

pangolin::GlBuffer *CsvDataLoader::GetVbo() {
  pangolin::GlBuffer *vbo =
      new pangolin::GlBuffer(pangolin::GlArrayBuffer, this->ptc_size, GL_FLOAT,
                             this->vertex_size, GL_DYNAMIC_DRAW);
  vbo->Upload(static_cast<void *>(this->vertices), vbo->size_bytes, 0);

  vbo->Bind();

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
  vbo->Unbind();
  LOG(INFO) << vbo->SizeBytes() << " size bytes";

  return vbo;
}

MapLoader::MapLoader(std::string pcd_file, double sample_ratio,
                     double intensity_scalar, double rgb_scalar,
                     double surfel_size, bool load_rgb)
    : vertex_size(Vertex::SIZE / sizeof(float)) {
  if (pcd_file != "") {
    pcl_utils.LoadPcd(pcd_file, sample_ratio, false);

    if (!load_rgb) {
      PointCloud ptc_tmp = *(pcl_utils.GetPointCloud());

      PointCloud ptc;

      std::vector<int> indices;
      pcl::removeNaNFromPointCloud(ptc_tmp, ptc, indices);
      ptc_size = ptc.size();
      LOG(INFO) << "point cloud size: " << ptc_size;

      vertices = new float[ptc_size * vertex_size];

      for (int i = 0; i < ptc_size; ++i) {

        vertices[i * vertex_size] = ptc[i].x;
        vertices[i * vertex_size + 1] = ptc[i].y;
        vertices[i * vertex_size + 2] = ptc[i].z;
        vertices[i * vertex_size + 3] = 0;
        vertices[i * vertex_size + 4] = 0;
        vertices[i * vertex_size + 5] = -1;  // -1 color undefined, but will be
        // reset in init_unstable.vert
        vertices[i * vertex_size + 6] = 0;
        // WARNING: temporal use for intensity
        vertices[i * vertex_size + 7] =
            int(ptc[i].intensity * intensity_scalar);
        vertices[i * vertex_size + 8] = ptc[i].normal[0];
        vertices[i * vertex_size + 9] = ptc[i].normal[1];
        vertices[i * vertex_size + 10] = ptc[i].normal[2];
        vertices[i * vertex_size + 11] = surfel_size;
      }
    } else {
      RGBPointCloud ptc_tmp = *(pcl_utils.GetRGBPointCloud());
      RGBPointCloud ptc;

      std::vector<int> indices;
      pcl::removeNaNFromPointCloud(ptc_tmp, ptc, indices);
      ptc_size = ptc.size();
      LOG(INFO) << "point cloud size: " << ptc_size;

      vertices = new float[ptc_size * vertex_size];

      for (int i = 0; i < ptc_size; ++i) {
        int rgb = int(ptc[i].r);
        rgb = (rgb << 8) + int(ptc[i].g);
        rgb = (rgb << 8) + int(ptc[i].b);
        int gray = 0.299 * ptc[i].r + 0.587 * ptc[i].g + 0.114 * ptc[i].b;

        vertices[i * vertex_size] = ptc[i].x;
        vertices[i * vertex_size + 1] = ptc[i].y;
        vertices[i * vertex_size + 2] = ptc[i].z;
        vertices[i * vertex_size + 3] = 0;
        vertices[i * vertex_size + 4] = rgb;
        vertices[i * vertex_size + 5] = 1;  // -1 color undefined, but will be
        // reset in init_unstable.vert
        vertices[i * vertex_size + 6] = 0;
        // WARNING: temporal use for intensity
        vertices[i * vertex_size + 7] = gray;
        vertices[i * vertex_size + 8] = ptc[i].normal[0];
        vertices[i * vertex_size + 9] = ptc[i].normal[1];
        vertices[i * vertex_size + 10] = ptc[i].normal[2];
        vertices[i * vertex_size + 11] = surfel_size;
      }
    }
  }
}

MapLoader::~MapLoader() { delete[] vertices; }

pangolin::GlBuffer *MapLoader::GetVbo() {
  pangolin::GlBuffer *vbo =
      new pangolin::GlBuffer(pangolin::GlArrayBuffer, this->ptc_size, GL_FLOAT,
                             this->vertex_size, GL_DYNAMIC_DRAW);
  vbo->Upload(static_cast<void *>(this->vertices), vbo->size_bytes, 0);

  vbo->Bind();

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
  vbo->Unbind();
  LOG(INFO) << vbo->SizeBytes() << " size bytes";

  return vbo;
}

}  // namespace dsl