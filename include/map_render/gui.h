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

#ifndef DSL_GUI_H_
#define DSL_GUI_H_

#include <pangolin/pangolin.h>
#include <pangolin/gl/gl.h>
#include <pangolin/gl/gldraw.h>
#include <map>
#include "core/gpu_texture.h"
#include "util/global_calib.h"
#include "shaders/shaders.h"

#define GL_GPU_MEM_INFO_TOTAL_AVAILABLE_MEM_NVX 0x9048
#define GL_GPU_MEM_INFO_CURRENT_AVAILABLE_MEM_NVX 0x9049

namespace dsl {

class GUI {
 public:
  GUI(bool offscreen = false, bool no_panel = false);

  virtual ~GUI();

  void PreCall();

  void PostCall();

  void DisplayImg(const std::string &id, GPUTexture *img, bool flipy = false);

  inline void DrawFrustum(const Eigen::Matrix4f &pose, float scale = 0.1) {
    Eigen::Matrix3f K = Eigen::Matrix3f::Identity();
    K(0, 0) = fxG[0];
    K(1, 1) = fyG[0];
    K(0, 2) = cxG[0];
    K(1, 2) = cyG[0];

    Eigen::Matrix3f Kinv = K.inverse();

    pangolin::glDrawFrustum(Kinv,
                            wG[0],
                            hG[0],
                            pose,
                            scale);
  }

  void DrawAxes(const Eigen::Matrix4f &pose, float scale) {
    pangolin::glSetFrameOfReference(pose);

    const GLfloat x = scale * 1;
    const GLfloat y = scale * 1;
    const GLfloat z = scale * 1;

    const GLfloat x_verts[] = {
        0, 0, 0, x, 0, 0
    };

    glColor3f(1, 0, 0);

    pangolin::glDrawVertices(2, x_verts, GL_LINE_STRIP, 3);

    const GLfloat y_verts[] = {
        0, 0, 0, 0, y, 0
    };

    glColor3f(0, 1, 0);

    pangolin::glDrawVertices(2, y_verts, GL_LINE_STRIP, 3);

    const GLfloat z_verts[] = {
        0, 0, 0, 0, 0, z
    };

    glColor3f(0, 0, 1);

    pangolin::glDrawVertices(2, z_verts, GL_LINE_STRIP, 3);

    glColor3f(0, 0, 0);

    pangolin::glUnsetFrameOfReference();
  }

  void DrawWorldPoints(const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > &world_points,
                       float point_size = 1.0,
                       const Eigen::Vector3f &color = Eigen::Vector3f(1, 0, 0)) {
    Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
    pangolin::glSetFrameOfReference(pose);

    glColor3f(color.x(), color.y(), color.z());
    glPointSize(point_size);

    pangolin::glDrawPoints(world_points);

    glPointSize(1.0);

    glColor3f(0, 0, 0);

    pangolin::glUnsetFrameOfReference();
  }

  void DrawMeshes(const std::vector<std::vector<Eigen::Vector3f,
                                                Eigen::aligned_allocator<Eigen::Vector3f> > > &triangles,
                  const Eigen::Vector3f &color = Eigen::Vector3f(0, 0, 1)) {
    Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
    pangolin::glSetFrameOfReference(pose);

    glColor3f(color.x(), color.y(), color.z());

    for (const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > &triangle:triangles) {
      pangolin::glDrawLineLoop(triangle);
    }

    glColor3f(0, 0, 0);

    pangolin::glUnsetFrameOfReference();
  }

  void SetFollowing(const Eigen::Matrix4f &currPose);

  void FollowPose(const Eigen::Matrix4f &currPose);
  void FollowAbsPose(const Eigen::Matrix4f &currPose);

  int width;
  int height;
  int panel;

  pangolin::Var<bool> *pause,
      *debug,
      *tracking_debug,
      *step,
      *followPose,
      *save,
      *run_opt,
      *draw_normals,
      *draw_colors,
      *draw_predict,
      *draw_global_model,
      *draw_trajectory;
  pangolin::Var<int> *gpuMem;
  pangolin::Var<int> *dataIdx;
  pangolin::Var<float> *accuracyIdx;
  pangolin::Var<float> *percentageIdx;
  pangolin::Var<std::string> *totalPoints;

  pangolin::DataLog resLog, inLog;
  pangolin::Plotter *resPlot,
      *inPlot;

  pangolin::OpenGlRenderState s_cam;
  pangolin::OpenGlMatrix delta_mv;

  pangolin::GlRenderBuffer *renderBuffer;
  pangolin::GlFramebuffer *colorFrameBuffer;
  GPUTexture *colorTexture;
  std::shared_ptr<Shader> colorProgram;

};

}

#endif  // DSL_GUI_H_
