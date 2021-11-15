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

#include "map_render/pcl_utils.h"

namespace dsl {

PclUtils::PclUtils() : cloud_ptr_{new PointCloud()}, rgb_cloud_ptr_{new RGBPointCloud()} {}

void PclUtils::LoadPcd(std::string file_name, double random_sample, bool estimate_normal) {
  if (pcd_reader_.read(file_name, *cloud_ptr_) != -1) {
    LOG(INFO) << "load " << file_name << " succeeded";
    if (random_sample != 1) {
      pcl::RandomSample<PointT> sample(true);
      // NOTE: be deterministic?
      sample.setSeed(7);
      sample.setInputCloud(cloud_ptr_);
      PointCloud::Ptr filtered_ptr_(new PointCloud());

      sample.setSample((*cloud_ptr_).size() * random_sample);
      sample.filter(*filtered_ptr_);

      cloud_ptr_ = filtered_ptr_;
    }

    if (pcd_reader_.read(file_name, *rgb_cloud_ptr_) != -1) {
      LOG(INFO) << "load " << file_name << " succeeded";
      if (random_sample != 1) {
        pcl::RandomSample<RGBPointT> sample(true);
        // NOTE: be deterministic?
        sample.setSeed(7);
        sample.setInputCloud(rgb_cloud_ptr_);
        RGBPointCloud::Ptr filtered_ptr_(new RGBPointCloud());

        sample.setSample((*rgb_cloud_ptr_).size() * random_sample);
        sample.filter(*filtered_ptr_);

        rgb_cloud_ptr_ = filtered_ptr_;
      }

      if (estimate_normal) {
        pcl::NormalEstimationOMP<PointT, PointT> ne;
        ne.setInputCloud(cloud_ptr_);

        pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
        ne.setSearchMethod(tree);

        pcl::PointCloud<PointT>::Ptr cloud_normals(new pcl::PointCloud<PointT>);
        ne.setRadiusSearch(0.1);

        *cloud_normals = *cloud_ptr_;
        ne.compute(*cloud_normals);

        cloud_ptr_ = cloud_normals;
      }

    }
  }
}

void PclUtils::SavePcd(std::string file_name) {
  pcd_writer_.writeBinary(file_name, *cloud_ptr_);
}

void PclUtils::SaveRGBPcd(std::string file_name) {
  pcd_writer_.writeBinary(file_name, *rgb_cloud_ptr_);
}

void PclUtils::PangolinToPcd(pangolin::GlTexture *vertex_tex, pangolin::GlTexture *normal_tex) {
  LOG_ASSERT(pgl_utils_.SetVertexNormal(vertex_tex, normal_tex));

  cloud_ptr_ = pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>);
  PointT pcl_point;

  int width = pgl_utils_.width;
  int height = pgl_utils_.height;

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      if (pgl_utils_.GetPoint(x, y, pcl_point)) {
        cloud_ptr_->push_back(pcl_point);
      }
    }
  }
}

void PclUtils::GlobalModelToPcd(GlobalModel &global_model) {

  Eigen::Vector4f * mapData = global_model.DownloadMap();
  rgb_cloud_ptr_->clear();

  for(unsigned int i = 0; i <  global_model.GetCount(); i++)
  {
    Eigen::Vector4f pos = mapData[(i * 3) + 0];
    Eigen::Vector4f col = mapData[(i * 3) + 1];
    Eigen::Vector4f nor = mapData[(i * 3) + 2];
    RGBPointT rgb_point;
    rgb_point.x = pos.x();
    rgb_point.y = pos.y();
    rgb_point.z = pos.z();

    rgb_point.r = int(col[0]) >> 16 & 0xFF;;
    rgb_point.g = int(col[0]) >> 8 & 0xFF;
    rgb_point.b = int(col[0]) & 0xFF;

    rgb_point.normal_x = nor.x();
    rgb_point.normal_y = nor.y();
    rgb_point.normal_z = nor.z();

    rgb_cloud_ptr_->push_back(rgb_point);
  }

  delete[] mapData;
}

}