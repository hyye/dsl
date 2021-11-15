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

#include "map_render/gui.h"

namespace dsl {

GUI::GUI(bool offscreen, bool no_panel) {
  width = 1280;
  height = 720;
  panel = 205;

  if (no_panel) {
    panel = 0;
  }
  width += panel;

  pangolin::Params windowParams;

  if (offscreen) {
    windowParams = pangolin::Params({{"scheme", "headless"}});
  } else {
    windowParams.Set("SAMPLE_BUFFERS", 0);
    windowParams.Set("SAMPLES", 0);
  }

  pangolin::CreateWindowAndBind("Main", width, height, windowParams);
  pangolin::BindToContext("Main");

  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glPixelStorei(GL_PACK_ALIGNMENT, 1);

//  //Internally render at 3840x2160
//  renderBuffer = new pangolin::GlRenderBuffer(3840, 2160),
//      colorTexture = new GPUTexture(renderBuffer->width, renderBuffer->height, GL_RGBA32F, GL_LUMINANCE, GL_FLOAT, true);
//
//  colorFrameBuffer = new pangolin::GlFramebuffer;
//  colorFrameBuffer->AttachColour(*colorTexture->texture);
//  colorFrameBuffer->AttachDepth(*renderBuffer);
//
//  colorProgram = std::shared_ptr<Shader>(loadProgramFromFile("draw_global_surface.vert", "draw_global_surface_phong.frag", "draw_global_surface.geom"));
//  fxaaProgram = std::shared_ptr<Shader>(loadProgramFromFile("empty.vert", "fxaa.frag", "quad.geom"));

  glEnable(GL_DEPTH_TEST);
  glDepthMask(GL_TRUE);
  glDepthFunc(GL_LESS);

  s_cam = pangolin::OpenGlRenderState(pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.1, 1000),
                                      pangolin::ModelViewLookAt(0, 0, -1, 0, 0, 1, pangolin::AxisNegY));

  pangolin::Display("cam").SetBounds(0.0f, 1.0f, pangolin::Attach::Pix(panel), 1.0f, -640 / 480.0)
      .SetHandler(new pangolin::Handler3D(s_cam));

  pangolin::Display(GPUTexture::RGB).SetAspect(640.0f / 480.0f);

  pangolin::Display(GPUTexture::DEPTH_NORM).SetAspect(640.0f / 480.0f);

  pangolin::Display("ModelImg").SetAspect(640.0f / 480.0f);

  pangolin::Display("Model").SetAspect(640.0f / 480.0f);

  std::vector<std::string> labels;
  labels.push_back(std::string("residual"));
  labels.push_back(std::string("threshold"));
  resLog.SetLabels(labels);

  resPlot = new pangolin::Plotter(&resLog, 0, 300, 0, 0.0005, 30, 0.5);
  resPlot->Track("$i");

  std::vector<std::string> labels2;
  labels2.push_back(std::string("inliers"));
  labels2.push_back(std::string("threshold"));
  inLog.SetLabels(labels2);

  inPlot = new pangolin::Plotter(&inLog, 0, 300, 0, 40000, 30, 0.5);
  inPlot->Track("$i");
  if (!no_panel) {
    pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(panel));
  }
  pangolin::Display("multi").SetBounds(pangolin::Attach::Pix(0),
                                       1 / 4.0f,
                                       pangolin::Attach::Pix(panel),
                                       1.0)
      .SetLayout(pangolin::LayoutEqualHorizontal)
      .AddDisplay(pangolin::Display(GPUTexture::RGB))
      .AddDisplay(pangolin::Display("Model"))
      .AddDisplay(pangolin::Display(GPUTexture::DEPTH_NORM))
      .AddDisplay(pangolin::Display("ModelImg"))
      .AddDisplay(pangolin::Display("Padding"));
//      .AddDisplay(*resPlot)
//      .AddDisplay(*inPlot);

  pause = new pangolin::Var<bool>("ui.Pause", false, true);
  tracking_debug = new pangolin::Var<bool>("ui.Tracking Debug", false, true);
  step = new pangolin::Var<bool>("ui.Step", false, false);
  debug = new pangolin::Var<bool>("ui.Debug Step", false, false);

  followPose = new pangolin::Var<bool>("ui.Follow pose", false, true);
  totalPoints = new pangolin::Var<std::string>("ui.Total points", "0");

  draw_normals = new pangolin::Var<bool>("ui.Draw normals", false, true);
  draw_colors = new pangolin::Var<bool>("ui.Draw colors", true, true);

  draw_predict = new pangolin::Var<bool>("ui.Draw predict", true, true);
  draw_global_model = new pangolin::Var<bool>("ui.Draw global model", true, true);
  draw_trajectory = new pangolin::Var<bool>("ui.Draw trajectory", false, true);

  pangolin::RegisterKeyPressCallback(' ', pangolin::SetVarFunctor<bool>("ui.Step", true));
  pangolin::RegisterKeyPressCallback('d', pangolin::SetVarFunctor<bool>("ui.Debug", true));
  pangolin::RegisterKeyPressCallback('p', pangolin::ToggleVarFunctor("ui.Pause"));
  pangolin::RegisterKeyPressCallback('f', pangolin::ToggleVarFunctor("ui.Follow pose"));
  pangolin::RegisterKeyPressCallback('t', pangolin::ToggleVarFunctor("ui.Draw trajectory"));
  pangolin::RegisterKeyPressCallback('n', pangolin::ToggleVarFunctor("ui.Draw normals"));
  pangolin::RegisterKeyPressCallback('c', pangolin::ToggleVarFunctor("ui.Draw colors"));
  pangolin::RegisterKeyPressCallback('g', pangolin::ToggleVarFunctor("ui.Draw global model"));

  GLint total_mem_kb = 0;
  glGetIntegerv(GL_GPU_MEM_INFO_TOTAL_AVAILABLE_MEM_NVX,
                &total_mem_kb);
  gpuMem = new pangolin::Var<int>("ui.GPU memory free", 0, 0, total_mem_kb / 1024);
  dataIdx = new pangolin::Var<int>("ui.data index", 0, 0, 0);
  accuracyIdx = new pangolin::Var<float>("ui.pose accuracy", 0.0, 0.0, 0.1);
  percentageIdx = new pangolin::Var<float>("ui.percentage", 0.0, 0.0, 1.0);
  save = new pangolin::Var<bool>("ui.save", false, false);
  run_opt = new pangolin::Var<bool>("ui.run opt", false, false);

  delta_mv.SetIdentity();
}

GUI::~GUI() {
  delete pause;
  delete step;
  delete gpuMem;
  delete followPose;
  delete save;
  delete draw_normals;
  delete draw_colors;

  delete totalPoints;

  delete resPlot;
  delete inPlot;

//  delete renderBuffer;
//  delete colorFrameBuffer;
//  delete colorTexture;
}

void GUI::PreCall() {
  glClearColor(0.05, 0.05, 0.3, 0.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  width = pangolin::DisplayBase().v.w;
  height = pangolin::DisplayBase().v.h;

  pangolin::Display("cam").Activate(s_cam);
}

void GUI::PostCall() {
  GLint cur_avail_mem_kb = 0;
  glGetIntegerv(GL_GPU_MEM_INFO_CURRENT_AVAILABLE_MEM_NVX, &cur_avail_mem_kb);

  int memFree = cur_avail_mem_kb / 1024;

  gpuMem->operator=(memFree);

  pangolin::FinishFrame();

  glFinish();
}

void GUI::DisplayImg(const std::string &id, GPUTexture *img, bool flipy) {
  glDisable(GL_DEPTH_TEST);

  pangolin::Display(id).Activate();
  img->texture->RenderToViewport(flipy);

  glEnable(GL_DEPTH_TEST);
}

void GUI::SetFollowing(const Eigen::Matrix4f &currPose) {
  pangolin::OpenGlMatrix curr_mv = this->s_cam.GetModelViewMatrix();

  pangolin::OpenGlMatrix mv;

  Eigen::Matrix3f currRot = currPose.topLeftCorner(3, 3);

  Eigen::Quaternionf currQuat(currRot);
  Eigen::Vector3f forwardVector(0, 0, 1);
  Eigen::Vector3f upVector(0, -1, 0);

  Eigen::Vector3f forward = (currQuat * forwardVector).normalized();
  Eigen::Vector3f up = (currQuat * upVector).normalized();

  Eigen::Vector3f eye(currPose(0, 3), currPose(1, 3), currPose(2, 3));

  eye -= forward;

  Eigen::Vector3f at = eye + forward;

  // T_wc = [R_wc t_wc; 0 0 0 1];
  Eigen::Vector3f z = (eye - at).normalized();  // Forward
  Eigen::Vector3f x = up.cross(z).normalized(); // Right
  Eigen::Vector3f y = z.cross(x);

  // T_cw = [R_wc^T -R_wc^T*t_wc; 0 0 0 1];
  Eigen::Matrix4d m;
  m << x(0), x(1), x(2), -(x.dot(eye)),
      y(0), y(1), y(2), -(y.dot(eye)),
      z(0), z(1), z(2), -(z.dot(eye)),
      0, 0, 0, 1;

//  Eigen::Vector3d ypr = R2ypr(m.block(0, 0, 3, 3));
//  ypr.y() = 0;
//  ypr.z() = 0;
//  m.block(0, 0, 3, 3) = ypr2R(ypr);

  memcpy(&mv.m[0], m.data(), sizeof(Eigen::Matrix4d));

  delta_mv = curr_mv * mv.Inverse();
}

void GUI::FollowPose(const Eigen::Matrix4f &currPose) {
  pangolin::OpenGlMatrix mv;

  Eigen::Matrix3f currRot = currPose.topLeftCorner(3, 3);

  Eigen::Quaternionf currQuat(currRot);
  Eigen::Vector3f forwardVector(0, 0, 1);
  Eigen::Vector3f upVector(0, -1, 0);

  Eigen::Vector3f forward = (currQuat * forwardVector).normalized();
  Eigen::Vector3f up = (currQuat * upVector).normalized();

  Eigen::Vector3f eye(currPose(0, 3), currPose(1, 3), currPose(2, 3));

  eye -= forward;

  Eigen::Vector3f at = eye + forward;

  // T_wc = [R_wc t_wc; 0 0 0 1];
  Eigen::Vector3f z = (eye - at).normalized();  // Forward
  Eigen::Vector3f x = up.cross(z).normalized(); // Right
  Eigen::Vector3f y = z.cross(x);

  // T_cw = [R_wc^T -R_wc^T*t_wc; 0 0 0 1];
  Eigen::Matrix4d m;
  m << x(0), x(1), x(2), -(x.dot(eye)),
      y(0), y(1), y(2), -(y.dot(eye)),
      z(0), z(1), z(2), -(z.dot(eye)),
      0, 0, 0, 1;

//  Eigen::Vector3d ypr = R2ypr(m.block(0, 0, 3, 3));
//  ypr.y() = 0;
//  ypr.z() = 0;
//  m.block(0, 0, 3, 3) = ypr2R(ypr);

  memcpy(&mv.m[0], m.data(), sizeof(Eigen::Matrix4d));

  mv = delta_mv * mv;

  this->s_cam.SetModelViewMatrix(mv);
}

void GUI::FollowAbsPose(const Eigen::Matrix4f &currPose) {
  pangolin::OpenGlMatrix mv;

  Eigen::Matrix3f currRot = currPose.topLeftCorner(3, 3);

  Eigen::Quaternionf currQuat(currRot);
  Eigen::Vector3f forwardVector(0, 0, 1);
  Eigen::Vector3f upVector(0, -1, 0);

  Eigen::Vector3f forward = (currQuat * forwardVector).normalized();
  Eigen::Vector3f up = (currQuat * upVector).normalized();

  Eigen::Vector3f eye(currPose(0, 3), currPose(1, 3), currPose(2, 3));

  eye -= forward;

  Eigen::Vector3f at = eye + forward;

  // T_wc = [R_wc t_wc; 0 0 0 1];
  Eigen::Vector3f z = (eye - at).normalized();  // Forward
  Eigen::Vector3f x = up.cross(z).normalized(); // Right
  Eigen::Vector3f y = z.cross(x);

  // T_cw = [R_wc^T -R_wc^T*t_wc; 0 0 0 1];
  Eigen::Matrix4d m;
  m << x(0), x(1), x(2), -(x.dot(eye)),
      y(0), y(1), y(2), -(y.dot(eye)),
      z(0), z(1), z(2), -(z.dot(eye)),
      0, 0, 0, 1;

  memcpy(&mv.m[0], m.data(), sizeof(Eigen::Matrix4d));

  this->s_cam.SetModelViewMatrix(mv);
}

}