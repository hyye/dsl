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

#include "optimization/homo_cost_functor.h"
#include "full_system/hessian_blocks.h"
#include "full_system/residual_projection.h"

namespace dsl {

double Distance2d(double x1, double y1, double x2, double y2) {
  double x = x1 - x2;
  double y = y1 - y2;
  return sqrt(x * x + y * y);
}

// TODO: combine cost functor with residual ???
HomoCostFunctor::HomoCostFunctor(PointFrameResidual *residual,
                                 dsl::CalibHessian &HCalib, bool baseline)
    : PhotoCostFunctor(residual, HCalib, baseline) {
  pfr_->ph_idist_backup = point_->idist;
}

bool HomoCostFunctor::Evaluate(double const *const *parameters,
                               double *residuals, double **jacobians) const {
  Eigen::Map<VecNR> r(residuals);
  Eigen::Map<const Vec6> param0(parameters[0]);
  Eigen::Map<const Vec2> param1(parameters[1]);
  Eigen::Map<const Vec6> param2(parameters[2]);
  Eigen::Map<const Vec2> param3(parameters[3]);

  SE3 T_t = SE3::exp(param0);
  Vec2 ab_t = param1;
  SE3 T_h = SE3::exp(param2);
  Vec2 ab_h = param3;

  if (pfr_->valid_plane && !baseline_) {
    {
      if (point_->convereged_ph_idist && !pfr_->idist_converged) {
        pfr_->idist_converged = true;
      }
    }

    pfr_->new_res_energy_with_outiler = -1;

    if (pfr_->res_state == ResState::OOB) {
      pfr_->new_res_state = ResState::OOB;
      SetEmpty(residuals, jacobians);
      r.setConstant(sqrt(pfr_->res_energy / 8.0));
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

    Eigen::Vector4f plane_coeff = pfr_->plane_coeff;
    Eigen::Vector4f plane_coeff_h = Eigen::Vector4f::Zero();
    // LOG(INFO) << "plane_coeff: " << plane_coeff.transpose();

    // WARNING: bug fixed here?
    plane_coeff_h =
        T_h.inverse().matrix().transpose().cast<float>() * plane_coeff;
    plane_coeff_h = (plane_coeff_h / (plane_coeff_h.head<3>().norm())).eval();
    const Eigen::Vector3f &n_h = plane_coeff_h.head<3>();
    const float &d_h = plane_coeff_h.w();
    const Mat33f &PRE_H = PRE_RTll_0 - PRE_tTll_0 * n_h.transpose() / d_h;

    float drescale, tc_x, tc_y, new_idist;
    float Ku, Kv;  // target pixel coordinate
    Vec3f KliP;    // host camera plane coordinate
    Vec3f ps_t;

    float Ku_0, Kv_0, new_id_0;
    Vec3f KliP_0;
    Vec3f ps_t_0;
    bool photo_valid = ProjectPointFisheye(
        point_->u, point_->v, pfr_->ph_idist_backup, 0, 0, HCalib_, PRE_RTll_0,
        PRE_tTll_0, drescale, ps_t_0, Ku_0, Kv_0, KliP_0, new_id_0);

    bool homo_valid = ProjectHomoPointFisheye(
        point_->u, point_->v, n_h, d_h, 0, 0, HCalib_, PRE_RTll_0, PRE_tTll_0,
        idist, drescale, ps_t, Ku, Kv, KliP, new_idist);

    if (!pfr_->idist_converged && photo_valid && homo_valid) {
      double pix_dis = Distance2d(Ku_0, Kv_0, Ku, Kv);
      float idist_ratio = fabs(1.0 / pfr_->ph_idist_backup * idist);
      if (idist_ratio > 1) {
        idist_ratio = 1.0 / idist_ratio;
      }

      // LOG(ERROR) << "@#@#@# SET ???????"
      //            << "pix_dis: " << pix_dis << " idist_ratio:" << idist_ratio;

      if (pix_dis >= 5 || fabs(1.0 - idist_ratio) > 0.5 || 1.0 / idist > 100) {
        SetOOB(residuals, jacobians, pfr_);
        return true;
      }

      // 3 pixel
      if (pix_dis < 2 && fabs(1.0 - idist_ratio) < 0.2) {
        pfr_->idist_converged = true;

        // LOG(INFO) << "@#@#@# SET CONVERGED!!!";
        // point->plane_valid = true;
        // point->scale_ratio = 1.0 / point->idist * idist;
      }
    } else {
      // LOG(ERROR) << "@#@#@# SET ???????; "
      //            << "photo: " << (photo_valid ? "yes" : "no")
      //            << "; homo: " << (homo_valid ? "yes" : "no");
    }

    if (!photo_valid || !homo_valid) {
      SetOOB(residuals, jacobians, pfr_);
      return true;
    }

    if (pfr_->idist_converged) {
      // LOG(INFO) << "converged!!!";
      const float *const color = point_->color;
      const float *const weights = point_->weights;
      Vec2f ab_th =
          AffLight::FromToVecExposure(host_->exposure, target_->exposure,
                                      AffLight(ab_h.x(), ab_h.y()),
                                      AffLight(ab_t.x(), ab_t.y()))
              .cast<float>();

      Vec2f ab_th0 = AffLight::FromToVecExposure(
                         host_->exposure, target_->exposure,
                         host_->GetAffLightZero(), target_->GetAffLightZero())
                         .cast<float>();
      float b0 = host_->GetAffLightZero().b;

      // float b0 = ab_h.y();

      // derivatives of pixel
      float du_didist, dv_didist;
      Eigen::Matrix<float, 2, 6> dp_dxi;
      Eigen::Matrix<float, 2, 6> dpn_dxih;
      Eigen::Matrix<float, 2, 3> dpt_dptps;
      Eigen::Matrix<float, 3, 3> dptps_dptps_hat;

      pfr_->ph_idist_backup = idist;
      pfr_->center_projected_to_backup = Vec3f(Ku, Kv, new_idist);
      if (first_eval) {
        pfr_->ph_idist = pfr_->ph_idist_backup;
        pfr_->center_projected_to = pfr_->center_projected_to_backup;
        first_eval = false;
      }

      if (jacobians != nullptr) {
        float scale_nxd = n_h.dot(KliP) / d_h;

        float xs = ps_t.x();
        float ys = ps_t.y();
        float zs = ps_t.z();
        float zs_xi_inv = 1.0 / (zs + HCalib_.xil());
        float zs_xi_invSq = zs_xi_inv * zs_xi_inv;
        //        const Vec3f &t = PRE_tTll_0;

        float xs2 = xs * xs;
        float ys2 = ys * ys;
        float zs2 = zs * zs;
        float oneMinusXs2 = 1 - xs2;
        float oneMinusYs2 = 1 - ys2;
        float oneMinusZs2 = 1 - zs2;
        float xsys = xs * ys;
        float yszs = ys * zs;
        float xszs = xs * zs;

        dpt_dptps << HCalib_.fxl() * zs_xi_inv, 0,
            -HCalib_.fxl() * xs * zs_xi_invSq, 0, HCalib_.fyl() * zs_xi_inv,
            -HCalib_.fyl() * ys * zs_xi_invSq;
        dptps_dptps_hat << oneMinusXs2, -xsys, -xszs, -xsys, oneMinusYs2, -yszs,
            -xszs, -yszs, oneMinusZs2;
        dptps_dptps_hat *= drescale;
        dpt_dptps *= dptps_dptps_hat;

        dp_dxi.block(0, 0, 2, 3) =
            dpt_dptps * -scale_nxd * Mat33f::Identity();  // dpdupsilon
        dp_dxi.block(0, 3, 2, 3) =
            dpt_dptps * (-SO3f::hat(PRE_RTll_0 * KliP) +
                         scale_nxd * SO3f::hat(PRE_tTll_0));  // dpdomega

        // xih --> xi^h_w
        dpn_dxih.block(0, 0, 2, 3) =
            dpt_dptps * -PRE_tTll_0 * scale_nxd / d_h * n_h.transpose();
        dpn_dxih.block(0, 3, 2, 3) =
            dpt_dptps * PRE_tTll_0 * KliP.transpose() / d_h * SO3f::hat(n_h);

        // WARNING: fix calib
      }

      float wJI2_sum = 0;

      VecNRf resF;

      VecNRf dIdu, dIdv;
      VecNRf dIda, dIdb;
      dIdu.setZero(), dIdv.setZero();
      dIda.setZero(), dIdb.setZero();

      for (int idx = 0; idx < patternNum; idx++) {
        float Ku, Kv;
        if (!ProjectHomoPointFisheye(point_->u + patternP[idx][0],
                                     point_->v + patternP[idx][1], n_h, d_h,
                                     HCalib_, PRE_RTll, PRE_tTll, Ku, Kv)) {
          SetOOB(residuals, jacobians, pfr_);
          return true;
        }

        pfr_->projected_to[idx][0] = Ku;
        pfr_->projected_to[idx][1] = Kv;

        Vec3f hit_color = (GetInterpolatedElement33(dIl, Ku, Kv, wG[0]));
        float residual =
            hit_color[0] - (float)(ab_th[0] * color[idx] + ab_th[1]);

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

        wJI2_sum += hw * hw *
                    (hit_color[1] * hit_color[1] + hit_color[2] * hit_color[2]);
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
        Eigen::Matrix<float, 2, 6> duvdxi, duvdxih;
        Eigen::Matrix<double, 8, 6, Eigen::RowMajor> Jdr_dxith;
        Eigen::Matrix<double, 2, 2, Eigen::RowMajor> Jdabth_dabt, Jdabth_dabh;

        dIduv.leftCols(1) = dIdu;
        dIduv.rightCols(1) = dIdv;
        duvdxi = dp_dxi;
        duvdxih = dpn_dxih;
        dIdab.leftCols(1) = dIda;
        dIdab.rightCols(1) = dIdb;

        Jdr_dxith = (dIduv * duvdxi).cast<double>();

        Jdabth_dabt << ab_th0.x(), 0, -ab_th0.x() * host_->GetAffLightZero().b,
            1;
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
          j = Jdr_dxith * -T_th0.Adj().cast<double>() +
              (dIduv * duvdxih).cast<double>();
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
          j.setZero();
          pfr_->Jd_backup = pfr_->Jd = j;
          // LOG(INFO) << "J4: " << j;
        }
      }

      // LOG(INFO) << energy_left << " - " << r.transpose();
      SetResidualFromNew(pfr_);
      return true;
    }
  }

  PhotoCostFunctor::Evaluate(parameters, residuals, jacobians);

  return true;
}

}  // namespace dsl
