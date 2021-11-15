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

#ifndef DSL_PCL_UTILS_H_
#define DSL_PCL_UTILS_H_

#include <pcl/filters/filter.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <pcl/filters/random_sample.h>
//#include <pcl/features/normal_3d.h>
#include <pangolin/pangolin.h>
#include <pcl/features/normal_3d_omp.h>
#include "core/global_model.h"
#include "pangolin_utils.h"

namespace dsl {

typedef pcl::PointXYZINormal PointT;
typedef pcl::PointXYZRGBNormal RGBPointT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef pcl::PointCloud<RGBPointT> RGBPointCloud;

class PclUtils {
 public:
  PclUtils();
  void LoadPcd(std::string file_name, double random_sample = 1,
               bool estimate_normal = false);
  void SavePcd(std::string file_name);
  void SaveRGBPcd(std::string file_name);
  PointCloud::Ptr &GetPointCloud() { return cloud_ptr_; };
  RGBPointCloud::Ptr &GetRGBPointCloud() { return rgb_cloud_ptr_; };
  void SetPointCloud(PointCloud::Ptr &in_ptr) { cloud_ptr_ = in_ptr; };
  void PangolinToPcd(pangolin::GlTexture *vertex_tex,
                     pangolin::GlTexture *normal_tex = NULL);
  void GlobalModelToPcd(GlobalModel &global_model);

 private:
  pcl::PCDReader pcd_reader_;
  pcl::PCDWriter pcd_writer_;
  PointCloud::Ptr cloud_ptr_;
  RGBPointCloud::Ptr rgb_cloud_ptr_;
  PangolinUtils pgl_utils_;
};

}  // namespace dsl

#endif  // DSL_PCL_UTILS_H_
