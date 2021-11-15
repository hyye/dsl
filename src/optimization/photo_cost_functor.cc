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
// Created by hyye on 11/18/19.
//

#include "optimization/photo_cost_functor.h"
#include "full_system/hessian_blocks.h"
#include "full_system/residual_projection.h"

namespace dsl {

// FIXME: frame_energy_th, SetNewFrameEnergyTh

void PhotoCostFunctor::SetResidualFromNew(PointFrameResidual *pfr) const {
  if (pfr->res_state == ResState::OOB) {
    return;
  }
  if (pfr->new_res_state == ResState::IN) {
    pfr->ef_residual->is_active_and_is_good_new = true;
  } else {
    pfr->ef_residual->is_active_and_is_good_new = false;
  }
  pfr->SetState(pfr->new_res_state);
  pfr->res_energy = pfr->new_res_energy;
  pfr->res_raw = pfr->new_res_raw;
}

void PhotoCostFunctor::SetOOB(double *residuals, double **jacobians,
                              PointFrameResidual *pfr) const {
  Eigen::Map<VecNR> r(residuals);
  pfr_->new_res_state = ResState::OOB;
  SetEmpty(residuals, jacobians);
  r.setConstant(sqrt(pfr_->res_energy / 8.0));
  pfr_->new_res_energy = pfr_->res_energy;
  pfr_->new_res_raw = pfr_->res_raw = r.dot(r);
  SetResidualFromNew(pfr);
}

void PhotoCostFunctor::SetEmpty(double *residuals, double **jacobians) const {
  Eigen::Map<VecNR> r(residuals);
  r.setZero();
  if (jacobians != nullptr) {
    if (jacobians[0] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 8, 6, Eigen::RowMajor>> j(jacobians[0]);
      j.setZero();
    }
    if (jacobians[1] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 8, 2, Eigen::RowMajor>> j(jacobians[1]);
      j.setZero();
    }
    if (jacobians[2] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 8, 6, Eigen::RowMajor>> j(jacobians[2]);
      j.setZero();
    }
    if (jacobians[3] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 8, 2, Eigen::RowMajor>> j(jacobians[3]);
      j.setZero();
    }
    if (jacobians[4] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 8, 1>> j(jacobians[4]);
      j.setZero();
    }
  }
  SetResidualFromNew(pfr_);
}

double PhotoCostFunctor::CalcHdd(const SE3f &host_to_target, double idist) {
  float Ku, Kv;
  Vec3f KliP;
  Vec3f ps_t;
  float drescale, new_idist;
  if (!ProjectPointFisheye(point_->u, point_->v, idist, 0, 0, HCalib_,
                           host_to_target.rotationMatrix(),
                           host_to_target.translation(), drescale, ps_t, Ku, Kv,
                           KliP, new_idist)) {
    return 1e-10;
  }
  float du_didist, dv_didist;
  float xs = ps_t.x();
  float ys = ps_t.y();
  float zs = ps_t.z();
  float zs_xi_inv = 1.0 / (zs + HCalib_.xil());
  float zs_xi_invSq = zs_xi_inv * zs_xi_inv;
  const Vec3f &t = host_to_target.translation();

  // diff d_idist
  du_didist =
      zs_xi_invSq * drescale *
      ((zs * t.x() - xs * t.z() + t.x() * HCalib_.xil() -
        xs * xs * t.x() * HCalib_.xil() - xs * ys * t.y() * HCalib_.xil() -
        xs * zs * t.z() * HCalib_.xil())) *
      SCALE_IDIST * HCalib_.fxl();
  dv_didist =
      zs_xi_invSq * drescale *
      ((zs * t.y() - ys * t.z() + t.y() * HCalib_.xil() -
        ys * ys * t.y() * HCalib_.xil() - ys * zs * t.z() * HCalib_.xil() -
        xs * ys * t.x() * HCalib_.xil())) *
      SCALE_IDIST * HCalib_.fyl();
}

// TODO: combine cost functor with residual ???
PhotoCostFunctor::PhotoCostFunctor(PointFrameResidual *residual,
                                   dsl::CalibHessian &HCalib, bool baseline)
    : point_(residual->point),
      host_(residual->host),
      target_(residual->target),
      pfr_(residual),
      HCalib_(HCalib),
      baseline_(baseline) {
  pfr_->Jd.setConstant(0);
  pfr_->Jd_backup.setConstant(0);
}

bool PhotoCostFunctor::Evaluate(double const *const *parameters,
                                double *residuals, double **jacobians) const {
  Eigen::Map<VecNR> r(residuals);
  Eigen::Map<const Vec6> param0(parameters[0]);
  Eigen::Map<const Vec2> param1(parameters[1]);
  Eigen::Map<const Vec6> param2(parameters[2]);
  Eigen::Map<const Vec2> param3(parameters[3]);
  const double &idist = parameters[4][0];
  SE3 T_t = SE3::exp(param0);
  Vec2 ab_t = param1;
  SE3 T_h = SE3::exp(param2);
  Vec2 ab_h = param3;

  if (!pfr_->idist_converged) {
    pfr_->new_res_energy_with_outiler = -1;

    if (pfr_->res_state == ResState::OOB) {
      pfr_->new_res_state = ResState::OOB;
      SetEmpty(residuals, jacobians);
      r.setConstant(sqrt(pfr_->res_energy / 8.0));
      pfr_->new_res_energy = pfr_->res_energy;
      pfr_->new_res_raw = pfr_->res_raw = r.dot(r);
      return true;
    }

    FrameFramePrecalc *precalc = &(host_->target_precalc[target_->idx]);
    float energy_left = 0;
    const Eigen::Vector3f *dIl = target_->dI;
    // const Mat33f& PRE_RTll_0 = precalc->PRE_RTll_0;
    // const Vec3f& PRE_tTll_0 = precalc->PRE_tTll_0;
    // const Mat33f& PRE_RTll = precalc->PRE_RTll;  // FIXME: debug PRE_RTll
    // const Vec3f& PRE_tTll = precalc->PRE_tTll;   // FIXME: debug PRE_tTll

    SE3f T_th = (T_t * T_h.inverse()).cast<float>();
    SE3f T_th0 = (target_->GetWorldToCamEvalPT() *
                  host_->GetWorldToCamEvalPT().inverse())
                     .cast<float>();
    const Mat33f &PRE_RTll_0 = T_th0.rotationMatrix();
    const Vec3f &PRE_tTll_0 = T_th0.translation();
    const Mat33f &PRE_RTll = T_th.rotationMatrix();
    const Vec3f &PRE_tTll = T_th.translation();

    const float *const color = point_->color;
    const float *const weights = point_->weights;
    Vec2f ab_th = AffLight::FromToVecExposure(host_->exposure, target_->exposure,
                                        AffLight(ab_h.x(), ab_h.y()),
                                        AffLight(ab_t.x(), ab_t.y()))
                .cast<float>();

    Vec2f ab_th0 = AffLight::FromToVecExposure(
                       host_->exposure, target_->exposure,
                       host_->GetAffLightZero(), target_->GetAffLightZero())
                       .cast<float>();
    float b0 =  host_->GetAffLightZero().b;
    // float b0 = ab_h.y();

    // derivatives of pixel
    Vec6f du_dxi, dv_dxi;
    Vec4f d_C_x, d_C_y;
    float du_didist, dv_didist;

    float drescale, new_idist;
    float Ku, Kv;
    Vec3f KliP;
    Vec3f ps_t;

    if (!ProjectPointFisheye(point_->u, point_->v, idist, 0, 0, HCalib_,
                             PRE_RTll_0, PRE_tTll_0, drescale, ps_t, Ku, Kv,
                             KliP, new_idist)) {
      SetOOB(residuals, jacobians, pfr_);
      return true;
    }

    pfr_->center_projected_to_backup = Vec3f(Ku, Kv, new_idist);
    if (first_eval) {
      pfr_->center_projected_to = pfr_->center_projected_to_backup;
      first_eval = false;
    }

    if (jacobians != nullptr) {
      float xs = ps_t.x();
      float ys = ps_t.y();
      float zs = ps_t.z();
      float zs_xi_inv = 1.0 / (zs + HCalib_.xil());
      float zs_xi_invSq = zs_xi_inv * zs_xi_inv;
      const Vec3f &t = PRE_tTll_0;

      // diff d_idist
      du_didist =
          zs_xi_invSq * drescale *
          ((zs * t.x() - xs * t.z() + t.x() * HCalib_.xil() -
            xs * xs * t.x() * HCalib_.xil() - xs * ys * t.y() * HCalib_.xil() -
            xs * zs * t.z() * HCalib_.xil())) *
          SCALE_IDIST * HCalib_.fxl();
      dv_didist =
          zs_xi_invSq * drescale *
          ((zs * t.y() - ys * t.z() + t.y() * HCalib_.xil() -
            ys * ys * t.y() * HCalib_.xil() - ys * zs * t.z() * HCalib_.xil() -
            xs * ys * t.x() * HCalib_.xil())) *
          SCALE_IDIST * HCalib_.fyl();

      // WARNING: fix calib

      float xs2 = xs * xs;
      float ys2 = ys * ys;
      float oneMinusXs2 = 1 - xs2;
      float oneMinusYs2 = 1 - ys2;
      float onePlusZsxi = 1 + zs * HCalib_.xil();
      float zsPlusXi = zs + HCalib_.xil();
      float xsys = xs * ys;

      du_dxi[0] = new_idist * zs_xi_invSq * (oneMinusXs2 * HCalib_.xil() + zs) *
                  HCalib_.fxl();
      du_dxi[1] =
          new_idist * zs_xi_invSq * (-xsys * HCalib_.xil()) * HCalib_.fxl();
      du_dxi[2] = new_idist * zs_xi_invSq * (-xs * onePlusZsxi) * HCalib_.fxl();
      du_dxi[3] = zs_xi_invSq * (-xsys) * HCalib_.fxl();
      du_dxi[4] = zs_xi_invSq * (onePlusZsxi - ys2) * HCalib_.fxl();
      du_dxi[5] = zs_xi_invSq * (-ys * zsPlusXi) * HCalib_.fxl();

      dv_dxi[0] =
          new_idist * zs_xi_invSq * (-xsys * HCalib_.xil()) * HCalib_.fyl();
      dv_dxi[1] = new_idist * zs_xi_invSq * (oneMinusYs2 * HCalib_.xil() + zs) *
                  HCalib_.fyl();
      dv_dxi[2] = new_idist * zs_xi_invSq * (-ys * onePlusZsxi) * HCalib_.fyl();
      dv_dxi[3] = zs_xi_invSq * (xs2 - onePlusZsxi) * HCalib_.fyl();
      dv_dxi[4] = zs_xi_invSq * (xsys)*HCalib_.fyl();
      dv_dxi[5] = zs_xi_invSq * (xs * zsPlusXi) * HCalib_.fyl();
    }

    float wJI2_sum = 0;

    VecNRf resF;

    VecNRf dIdu, dIdv;
    VecNRf dIda, dIdb;
    dIdu.setZero(), dIdv.setZero();
    dIda.setZero(), dIdb.setZero();

    for (int idx = 0; idx < patternNum; idx++) {
      float Ku, Kv;
      if (!ProjectPointFisheye(point_->u + patternP[idx][0],
                               point_->v + patternP[idx][1], idist, HCalib_,
                               PRE_RTll, PRE_tTll, Ku, Kv)) {
        SetOOB(residuals, jacobians, pfr_);
        return true;
      }

      pfr_->projected_to[idx][0] = Ku;
      pfr_->projected_to[idx][1] = Kv;

      Vec3f hit_color = (GetInterpolatedElement33(dIl, Ku, Kv, wG[0]));
      float residual = hit_color[0] - (float)(ab_th[0] * color[idx] + ab_th[1]);

      float drdA = (color[idx]);
      if (!std::isfinite((float)hit_color[0])) {
        SetOOB(residuals, jacobians, pfr_);
        return true;
      }

      float w = sqrtf(
          settingOutlierThSumComponent /
          (settingOutlierThSumComponent + hit_color.tail<2>().squaredNorm()));
      w = 0.5f * (w + weights[idx]);

      float hw = fabsf(residual) < settingHuberTh
                     ? 1
                     : settingHuberTh / fabsf(residual);
      energy_left += w * w * hw * residual * residual * (2 - hw);

      if (hw < 1) hw = sqrtf(hw * (2 - hw));
      hw = hw * w;

      hit_color[1] *= hw;
      hit_color[2] *= hw;

      resF[idx] = residual * hw;

      wJI2_sum +=
          hw * hw * (hit_color[1] * hit_color[1] + hit_color[2] * hit_color[2]);
      // LOG(INFO) << wJI2_sum;

      if (jacobians != nullptr) {
        dIdu[idx] = hit_color[1];
        dIdv[idx] = hit_color[2];
        dIda[idx] = -drdA * hw;
        dIdb[idx] = -hw;

        if (settingAffineOptModeA < 0) dIda[idx] = 0;
        if (settingAffineOptModeB < 0) dIdb[idx] = 0;
      }
    }

    pfr_->new_res_energy_with_outiler = energy_left;

    double energy_ratio = 1;
    double frame_energy_th =
        std::max<float>(host_->frame_energy_th, target_->frame_energy_th);

    if (energy_left > frame_energy_th || wJI2_sum < 2) {
      energy_left = frame_energy_th;
      energy_ratio = sqrt(frame_energy_th / energy_left);
      pfr_->new_res_state = ResState::OUTLIER;
    } else {
      pfr_->new_res_state = ResState::IN;
    }

    if (!std::isfinite(energy_ratio)) {
      energy_ratio = 1;
      assert(false);
    }

    pfr_->new_res_energy = energy_left;
    r = resF.cast<double>();
    pfr_->new_res_raw = r.dot(r);
    r *= energy_ratio;

    if (jacobians != nullptr) {
      Eigen::Matrix<float, MAX_RES_PER_POINT, 2> dIduv, dIdab;
      Eigen::Matrix<float, 2, 6> duvdxi;
      Eigen::Matrix<double, 8, 6, Eigen::RowMajor> Jdr_dxith;
      Eigen::Matrix<double, 2, 2, Eigen::RowMajor> Jdabth_dabt, Jdabth_dabh;

      dIduv.leftCols(1) = dIdu;
      dIduv.rightCols(1) = dIdv;
      duvdxi.topRows(1) = du_dxi.transpose();
      duvdxi.bottomRows(1) = dv_dxi.transpose();
      dIdab.leftCols(1) = dIda;
      dIdab.rightCols(1) = dIdb;

      Jdr_dxith = (dIduv * duvdxi).cast<double>();

      Jdabth_dabt << ab_th0.x(), 0, -ab_th0.x() * host_->GetAffLightZero().b, 1;
      Jdabth_dabh << -ab_th0.x(), 0, ab_th0.x() * host_->GetAffLightZero().b,
          -ab_th0.x();
      if (jacobians[0] != nullptr) {
        Eigen::Map<Eigen::Matrix<double, 8, 6, Eigen::RowMajor>> j(
            jacobians[0]);

        j = Jdr_dxith;
        // LOG(INFO) << "J0: " << j;
      }
      if (jacobians[1] != nullptr) {
        Eigen::Map<Eigen::Matrix<double, 8, 2, Eigen::RowMajor>> j(
            jacobians[1]);
        j = dIdab.cast<double>() * Jdabth_dabt;
        if (!settigNoScalingAtOpt) {
          j.leftCols(1) *= SCALE_A;
          j.rightCols(1) *= SCALE_B;
        }
        // LOG(INFO) << "J1: " << j;
      }
      if (jacobians[2] != nullptr) {
        Eigen::Map<Eigen::Matrix<double, 8, 6, Eigen::RowMajor>> j(
            jacobians[2]);
        j = Jdr_dxith * -T_th0.Adj().cast<double>();
      }
      if (jacobians[3] != nullptr) {
        Eigen::Map<Eigen::Matrix<double, 8, 2, Eigen::RowMajor>> j(
            jacobians[3]);
        j = dIdab.cast<double>() * Jdabth_dabh;
        if (!settigNoScalingAtOpt) {
          j.leftCols(1) *= SCALE_A;
          j.rightCols(1) *= SCALE_B;
        }
      }
      if (jacobians[4] != nullptr) {
        Eigen::Map<Eigen::Matrix<double, 8, 1>> j(jacobians[4]);
        j = (dIdu * du_didist + dIdv * dv_didist).cast<double>();
        pfr_->Jd_backup = j;
        // LOG(INFO) << "J4: " << j;
      }
    }

    // LOG(INFO) << energy_left << " - " << r.transpose();
  }
  SetResidualFromNew(pfr_);
  return true;
}

}  // namespace dsl