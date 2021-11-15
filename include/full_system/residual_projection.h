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
// Created by hyye on 11/16/19.
//

#ifndef DSL_RESIDUAL_PROJECTION_H_
#define DSL_RESIDUAL_PROJECTION_H_

#include "dsl_common.h"
#include "hessian_blocks.h"

namespace dsl {

EIGEN_STRONG_INLINE float DeriveIdist(const Vec3f &t, const Vec3f &ps_t,
                                      const float &dxInterp,
                                      const float &dyInterp,
                                      const float &drescale, const float &xi) {
  float Xs = ps_t.x();
  float Ys = ps_t.y();
  float Zs = ps_t.z();
  float zs_xi_inv = 1.0 / (Zs + xi);
  return zs_xi_inv * zs_xi_inv * drescale *
         (dxInterp *
              (Zs * t.x() - Xs * t.z() + t.x() * xi - Xs * Xs * t.x() * xi -
               Xs * Ys * t.y() * xi - Xs * Zs * t.z() * xi) +
          dyInterp *
              (Zs * t.y() - Ys * t.z() + t.y() * xi - Ys * Ys * t.y() * xi -
               Ys * Zs * t.z() * xi - Xs * Ys * t.x() * xi)) *
         SCALE_IDIST;
}

EIGEN_STRONG_INLINE bool ProjectPointFisheye(const float &u_pt,
                                             const float &v_pt,
                                             const float &idist,
                                             CalibHessian &HCalib,
                                             const Mat33f &R, const Vec3f &t,
                                             float &Ku, float &Kv) {
  Vec3f KliP = LiftToSphere(u_pt, v_pt, HCalib.fxli(), HCalib.fyli(),
                            HCalib.cxli(), HCalib.cyli(), HCalib.xil());

  Vec3f ptp = R * KliP + t * idist;
  Vec3f p_t_image = SpaceToPlane(ptp, HCalib.fxl(), HCalib.fyl(), HCalib.cxl(),
                                 HCalib.cyl(), HCalib.xil());
  Ku = p_t_image.x();
  Kv = p_t_image.y();
  return Ku > 1.1f && Kv > 1.1f && Ku < wM3G && Kv < hM3G &&
         ValidArea(maskG[0], Ku, Kv);
}

EIGEN_STRONG_INLINE bool ProjectPointFisheye(
    const float &u_pt, const float &v_pt, const float &idist, const int &dx,
    const int &dy, CalibHessian &HCalib, const Mat33f &R, const Vec3f &t,
    float &drescale, Vec3f &ps_t, float &Ku, float &Kv, Vec3f &KliP,
    float &new_idist) {
  KliP = LiftToSphere((u_pt + dx), (v_pt + dy), HCalib.fxli(), HCalib.fyli(),
                      HCalib.cxli(), HCalib.cyli(), HCalib.xil());

  Vec3f ptp = R * KliP + t * idist;
  ps_t = ptp.normalized();
  drescale = 1.0f / ptp.norm();
  new_idist = idist * drescale;

  if (!(ps_t.z() + HCalib.xil() > 0)) return false;

  Vec3f p_t_image = SpaceToPlane(ptp, HCalib.fxl(), HCalib.fyl(), HCalib.cxl(),
                                 HCalib.cyl(), HCalib.xil());

  Ku = p_t_image.x();
  Kv = p_t_image.y();
  // DLOG(INFO) << u_pt << " " << v_pt << " " << Ku << " " << Kv;

  // w-3, h-3
  return Ku > 1.1f && Kv > 1.1f && Ku < wM3G && Kv < hM3G &&
         ValidArea(maskG[0], Ku, Kv);
}

EIGEN_STRONG_INLINE bool ProjectHomoPointFisheye(
    const float &u_pt, const float &v_pt, const Vec3f &n_h, const float &d_h,
    CalibHessian &HCalib, const Mat33f &R, const Vec3f &t, float &Ku,
    float &Kv) {
  Vec3f KliP = LiftToSphere(u_pt, v_pt, HCalib.fxli(), HCalib.fyli(),
                            HCalib.cxli(), HCalib.cyli(), HCalib.xil());
  Vec3f ptp = R * KliP - t * n_h.dot(KliP) / d_h;
  Vec3f p_t_image = SpaceToPlane(ptp, HCalib.fxl(), HCalib.fyl(), HCalib.cxl(),
                                 HCalib.cyl(), HCalib.xil());
  Ku = p_t_image.x();
  Kv = p_t_image.y();
  return Ku > 1.1f && Kv > 1.1f && Ku < wM3G && Kv < hM3G &&
         ValidArea(maskG[0], Ku, Kv);
}

EIGEN_STRONG_INLINE bool ProjectHomoPointFisheye(
    const float &u_pt, const float &v_pt, const Vec3f &n_h, const float &d_h,
    const int &dx, const int &dy, CalibHessian &HCalib, const Mat33f &R,
    const Vec3f &t, float &idist, float &drescale, Vec3f &ps_t, float &Ku,
    float &Kv, Vec3f &KliP, float &new_idist) {
  //  const Mat33f Ri = R.transpose();
  //  const Vec3f ti = -Ri * t;

  KliP = LiftToSphere((u_pt + dx), (v_pt + dy), HCalib.fxli(), HCalib.fyli(),
                      HCalib.cxli(), HCalib.cyli(), HCalib.xil());

  idist = -n_h.dot(KliP) / d_h;

  // LOG(INFO) << n_h.transpose() << " " << d_h << " idist: " << idist;

  Vec3f ptp = R * KliP + t * idist;
  ps_t = ptp.normalized();
  drescale = 1.0f / ptp.norm();

  new_idist = idist * drescale;

  if (!(ps_t.z() + HCalib.xil() > 0) || !(idist > 0) || !std::isfinite(idist))
    return false;

  Vec3f p_t_image = SpaceToPlane(ptp, HCalib.fxl(), HCalib.fyl(), HCalib.cxl(),
                                 HCalib.cyl(), HCalib.xil());
  Ku = p_t_image.x();
  Kv = p_t_image.y();

  // w-3, h-3
  return Ku > 1.1f && Kv > 1.1f && Ku < wM3G && Kv < hM3G &&
         ValidArea(maskG[0], Ku, Kv);
}

}  // namespace dsl

#endif  // DSL_RESIDUAL_PROJECTION_H_
