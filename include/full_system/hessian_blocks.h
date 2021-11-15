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
// Created by hyye on 11/5/19.
//

#ifndef DSL_HESSIAN_BLOCKS_H_
#define DSL_HESSIAN_BLOCKS_H_

#include "dsl_common.h"
#include "ef_struct.h"
#include "immature_point.h"
#include "residual.h"
#include "util/frame_shell.h"
#include "util/global_calib.h"
#include "util/image_and_exposure.h"

namespace dsl {

// contains affine parameters as XtoWorld.
inline Vec2 AffFromTo(const Vec2 &from, const Vec2 &to) {
  return Vec2(from[0] / to[0], (from[1] - to[1]) / to[0]);
}

struct FrameFramePrecalc {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  FrameHessian *host;    // defines row
  FrameHessian *target;  // defines column

  // precalc values
  Mat33f PRE_RTll;
  Mat33f PRE_KRKiTll;
  Mat33f PRE_RKiTll;
  Mat33f PRE_RTll_0;

  Vec2f PRE_aff_mode;
  float PRE_b0_mode;

  Vec3f PRE_tTll;
  Vec3f PRE_KtTll;
  Vec3f PRE_tTll_0;

  float distance_ll;

  inline ~FrameFramePrecalc() {}
  inline FrameFramePrecalc() { host = target = 0; }
  void Set(FrameHessian *host, FrameHessian *target, CalibHessian &HCalib);
};

struct FrameHessian {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  static int instanceCounter;

  FrameShell *shell;
  EfFrame *ef_frame;
  std::vector<std::unique_ptr<PointHessian>> point_hessians;
  std::vector<std::unique_ptr<PointHessian>> point_hessians_marginalized;
  std::vector<std::unique_ptr<PointHessian>> point_hessians_out;
  std::vector<std::unique_ptr<ImmaturePoint>> immature_points;

  Vec3f *dI;
  std::array<std::vector<Vec3f>, PYR_LEVELS> dIp;
  std::array<std::vector<float>, PYR_LEVELS> abs_sq_grad;

  // WARNING: c++17 will do everything?
  std::vector<Vec4f, Eigen::aligned_allocator<Vec4f>> vertex_map;
  std::vector<Vec4f, Eigen::aligned_allocator<Vec4f>> normal_map;
  std::vector<FrameFramePrecalc> target_precalc;

  int frame_id;
  int idx;

  float frame_energy_th;
  float exposure;

  bool flagged_for_marginalization;

  SE3 world_to_cam_evalPT;
  // NOTE: in the original DSO, the states of pose are the delta values
  Vec10 state;  /// [0-5: world_to_cam-left_eps. 6-7: a,b]
  Vec10 state_zero;

  EIGEN_STRONG_INLINE const SE3 &GetWorldToCamEvalPT() const {
    return world_to_cam_evalPT;
  }

  EIGEN_STRONG_INLINE const Vec10 &GetState() const { return state; }
  EIGEN_STRONG_INLINE const Vec10 &GetStateZero() const { return state_zero; }
  inline AffLight GetAffLight() const {
    return AffLight(GetState()[6], GetState()[7]);
  }
  inline AffLight GetAffLightZero() const {
    return AffLight(GetStateZero()[6], GetStateZero()[7]);
  }

  PoseParameterBlockSpec parameter_pose;
  AbParameterBlockSpec parameter_ab;

  // precalc values
  SE3 PRE_world_to_cam;
  SE3 PRE_cam_to_world;

  FrameHessian();
  ~FrameHessian() { --instanceCounter; }

  inline Vec6 W2CLeftEps() const { return GetState().head<6>(); }

  inline void SetStateZero(const Vec10 &state_zero_in) {
    assert(state_zero_in.head<6>().squaredNorm() < 1e-20);
    this->state_zero = state_zero_in;
    // WARNING: no nullspace here
  }

  inline void SetState(const Vec10 &state_in) {
    this->state = state_in;

    // left product
    PRE_world_to_cam = SE3::exp(W2CLeftEps()) * GetWorldToCamEvalPT();
    PRE_cam_to_world = PRE_world_to_cam.inverse();
  }

  inline void SetState(const Vec6 &pose, const Vec2 &ab) {
    this->state.setZero();
    this->state[6] = ab.x();
    this->state[7] = ab.y();

    PRE_world_to_cam = SE3::exp(pose);
    PRE_cam_to_world = PRE_world_to_cam.inverse();
  }

  inline void SetEvalPT(const SE3 &world_to_cam_evalPT, const Vec10 &state_in) {
    this->world_to_cam_evalPT = world_to_cam_evalPT;
    SetState(state_in);
    SetStateZero(state_in);
  }

  inline void SetZeroEvalPT(const SE3 &world_to_cam_evalPT,
                            const AffLight &aff_light) {
    Vec10 initial_state = Vec10::Zero();
    initial_state[6] = aff_light.a;
    initial_state[7] = aff_light.b;
    this->world_to_cam_evalPT = world_to_cam_evalPT;
    SetState(initial_state);
    SetStateZero(this->GetState());
  }

  void MakeImages(float *color, CalibHessian *HCalib);
};

struct PointHessian {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  static int instanceCounter;

  // static values
  float color[MAX_RES_PER_POINT];    // colors in host frame
  float weights[MAX_RES_PER_POINT];  // host-weights for respective residuals.

  bool convereged_ph_idist = false;
  Vec2f pix;
  Vec3f p_sphere;  // FIXME
  float u, v;
  int idx;
  float energy_th;
  bool has_dist_prior;

  float idist_hessian;
  float max_rel_baseline;
  int num_good_residuals;

  float my_type;

  float idist;
  float idist_zero;
  float step;
  float step_backup;
  float idist_backup;

  Vec4f plane_coeff;
  bool valid_plane = false;

  IdistParameterBlockSpec parameter_idist;

  EfPoint *ef_point;
  FrameHessian *host;
  std::vector<std::unique_ptr<PointFrameResidual>> residuals;
  // contains information about residuals to the last two (!) frames. ([0] =
  // latest, [1] = the one before).
  std::pair<PointFrameResidual *, ResState> last_residuals[2];

  enum PhStatus { ACTIVE = 0, INACTIVE, OUTLIER, OOB, MARGINALIZED };
  PhStatus status;

  PointHessian(const ImmaturePoint &raw_point, CalibHessian &HCalib);
  inline virtual ~PointHessian() { --instanceCounter; }
  inline void SetPhStatus(PhStatus s) { status = s; }
  inline void SetIdist(float idist_in) { idist = idist_in; }
  inline void SetIdistZero(float idist_in) { idist_zero = idist_in; }

  virtual bool SetPlane(
      std::vector<Vec4f, Eigen::aligned_allocator<Vec4f>> &vertex_map,
      std::vector<Vec4f, Eigen::aligned_allocator<Vec4f>> &normal_map);

  inline bool IsOOB(const std::vector<FrameHessian *> &fhs_to_marg) const {
    int res_in_to_marg = 0;
    for (const std::unique_ptr<PointFrameResidual> &r : residuals) {
      if (r->res_state != ResState::IN) continue;
      for (FrameHessian *k : fhs_to_marg)
        if (r->target == k) res_in_to_marg++;
    }

    // if long lasting res contains too few active res after marg
    if ((int)residuals.size() >= settingMinGoodActiveResForMarg &&
        num_good_residuals > settingMinGoodResForMarg + 10 &&
        (int)residuals.size() - res_in_to_marg < settingMinGoodActiveResForMarg)
      return true;

    // if the latest projection is OOB
    if (last_residuals[0].second == ResState::OOB) return true;
    // if there is no sufficient residual
    if (residuals.size() < 2) return false;
    // if the latest two projections become outlier
    if (last_residuals[0].second == ResState::OUTLIER &&
        last_residuals[1].second == ResState::OUTLIER)
      return true;
    // all otherwise cases
    return false;
  }

  inline bool IsInlierNew() {
    return (int)residuals.size() >= settingMinGoodActiveResForMarg &&
           num_good_residuals >= settingMinGoodResForMarg;
  }
};

struct CalibHessian {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  static int instanceCounter;

  VecC value_zero;
  VecCf valuef;
  VecCf valuei;
  VecC value;
  VecC step;
  VecC step_backup;
  VecC value_backup;
  VecC value_minus_value_zero;

  float xi;

  double log_scale;
  double log_scale_backup;
  double log_scale_step;
  SE3 global_drift_pose;
  SE3 global_init_pose;

  inline ~CalibHessian() { --instanceCounter; }
  inline CalibHessian() {
    VecC initial_value = VecC::Zero();
    initial_value[0] = fxG[0];
    initial_value[1] = fyG[0];
    initial_value[2] = cxG[0];
    initial_value[3] = cyG[0];

    xi = xiG;

    SetValue(initial_value);
    value_zero = value;
    value_minus_value_zero.setZero();

    ++instanceCounter;
    for (int i = 0; i < 256; i++)
      Binv[i] = B[i] = i;  // set gamma function to identity

    log_scale = 0.0;
    log_scale_backup = 0.0;
    global_drift_pose = SE3(Eigen::Matrix4d::Identity());
  };

  inline float &xil() { return xi; }

  // normal mode: use the optimized parameters everywhere!
  inline float &fxl() { return valuef[0]; }
  inline float &fyl() { return valuef[1]; }
  inline float &cxl() { return valuef[2]; }
  inline float &cyl() { return valuef[3]; }
  inline float &fxli() { return valuei[0]; }
  inline float &fyli() { return valuei[1]; }
  inline float &cxli() { return valuei[2]; }
  inline float &cyli() { return valuei[3]; }

  inline float xil() const { return xi; }
  inline float fxl() const { return valuef[0]; }
  inline float fyl() const { return valuef[1]; }
  inline float cxl() const { return valuef[2]; }
  inline float cyl() const { return valuef[3]; }
  inline float fxli() const { return valuei[0]; }
  inline float fyli() const { return valuei[1]; }
  inline float cxli() const { return valuei[2]; }
  inline float cyli() const { return valuei[3]; }

  inline void SetValue(const VecC &_value) {
    this->value = _value;
    this->valuef = this->value.cast<float>();

    this->value_minus_value_zero = this->value - this->value_zero;
    this->valuei[0] = 1.0f / this->valuef[0];
    this->valuei[1] = 1.0f / this->valuef[1];
    this->valuei[2] = -this->valuef[2] / this->valuef[0];
    this->valuei[3] = -this->valuef[3] / this->valuef[1];
  };

  float Binv[256];
  float B[256];

  EIGEN_STRONG_INLINE float GetBGradOnly(float color) {
    int c = color + 0.5f;
    if (c < 5) c = 5;
    if (c > 250) c = 250;
    return B[c + 1] - B[c];
  }

  EIGEN_STRONG_INLINE float GetBInvGradOnly(float color) {
    int c = color + 0.5f;
    if (c < 5) c = 5;
    if (c > 250) c = 250;
    return Binv[c + 1] - Binv[c];
  }
};

}  // namespace dsl

#endif  // DSL_HESSIAN_BLOCKS_H_
