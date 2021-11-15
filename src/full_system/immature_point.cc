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
// Created by hyye on 11/6/19.
//

#include "full_system/immature_point.h"
#include "full_system/hessian_blocks.h"
#include "full_system/residual_projection.h"

namespace dsl {

// http://geomalgorithms.com/a07-_distance.html
float IdistTriangulation(const Vec3f& vh, const Vec3f& vt, const Mat33f& R_th,
                         const Vec3f t_th) {
  Vec3f vh_in_t = R_th * vh;
  Vec3f P0 = t_th;
  Vec3f Q0 = Vec3f::Zero();
  Vec3f u = vh_in_t;
  Vec3f v = vt;
  Vec3f w0 = P0 - Q0;
  float a = u.dot(u);
  float b = u.dot(v);
  float c = v.dot(v);
  float d = u.dot(w0);
  float e = v.dot(w0);

  float s1 = (b * e - c * d);
  float s2 = (a * c - b * b);
  float s = s1 / s2;

  return 1.0 / s;
}

void ApproxPixelCoordinate(const Vec3f& vh_approx, const float& bestU,
                           const float& bestV, const Mat33f& hostToFrame_R,
                           const Vec3f& hostToFrame_t, CalibHessian& HCalib,
                           float& idistBest, Vec3f& bestUV) {
  Vec3f vt_approx = LiftToSphere(bestU, bestV, HCalib.fxli(), HCalib.fyli(),
                                 HCalib.cxli(), HCalib.cyli(), HCalib.xil());
  float idist_tmp =
      IdistTriangulation(vh_approx, vt_approx, hostToFrame_R, hostToFrame_t);
  idistBest = std::isnan(idist_tmp) ? idistBest : idist_tmp;
  bestUV = SpaceToPlane(hostToFrame_R * vh_approx + hostToFrame_t * idistBest,
                        HCalib.fxl(), HCalib.fyl(), HCalib.cxl(), HCalib.cyl(),
                        HCalib.xil());
}

ImmaturePoint::ImmaturePoint(int _u, int _v, FrameHessian* _host, float type,
                             CalibHessian& HCalib)
    : u(_u),
      v(_v),
      host(_host),
      my_type(type),
      idist_min(0),
      idist_max(NAN),
      last_trace_status(ImmaturePointStatus::IPS_UNINITIALIZED) {
  Vec3f ps = LiftToSphere(u, v, HCalib.fxli(), HCalib.fyli(), HCalib.cxli(),
                          HCalib.cyli(), HCalib.xil());
  x = ps.x();
  y = ps.y();
  z = ps.z();
  gradH.setZero();

  /// NOTE: with bilinear interpolated color
  for (int idx = 0; idx < patternNum; idx++) {
    int dx = patternP[idx][0];
    int dy = patternP[idx][1];

    Vec3f ptc = GetInterpolatedElement33BiLin(host->dI, u + dx, v + dy, wG[0]);

    color[idx] = ptc[0];
    if (!std::isfinite(color[idx])) {
      energy_th = NAN;
      return;
    }

    gradH += ptc.tail<2>() * ptc.tail<2>().transpose();

    weights[idx] =
        sqrtf(settingOutlierThSumComponent /
              (settingOutlierThSumComponent + ptc.tail<2>().squaredNorm()));
  }
  energy_th = patternNum * settingOutlierTh;
  energy_th *= settingOverallEnergyThWeight * settingOverallEnergyThWeight;

  quality = 10000;
}

double ImmaturePoint::LinearizeResidual(CalibHessian& HCalib,
                                        const float outlier_th_slack,
                                        ImmaturePointTemporaryResidual& tmp_res,
                                        float& Hdd, float& bd, float idist) {
  /// will not happen from optimizeImmaturePoint
  if (tmp_res.res_state == ResState::OOB) {
    tmp_res.new_res_state = ResState::OOB;
    return tmp_res.res_energy;
  }

  /// from setPrecalcValues
  const FrameFramePrecalc& precalc =
      (host->target_precalc[tmp_res.target->idx]);

  // check OOB due to scale angle change.

  float energy_left = 0;
  const Vec3f* dIl = tmp_res.target->dI;
  const Mat33f& PRE_RTll = precalc.PRE_RTll;  /// from host to target (new)
  const Vec3f& PRE_tTll = precalc.PRE_tTll;
  // const float * const Il = tmp_res->target->I;

  Vec2f affLL = precalc.PRE_aff_mode;

  // LOG(INFO) << "affLL1: " << affLL.transpose();
  // LOG(INFO) << "affLL2: "
  //           << AffLight::FromToVecExposure(
  //                  host->exposure, tmp_res.target->exposure,
  //                  host->GetAffLight(), tmp_res.target->GetAffLight())
  //                  .transpose();

  for (int idx = 0; idx < patternNum; idx++) {
    int dx = patternP[idx][0];
    int dy = patternP[idx][1];

    float drescale, u, v, new_idist;
    float Ku, Kv;
    Vec3f KliP;
    Vec3f ps_t;

    /// project into the new frame
    if (!ProjectPointFisheye(this->u, this->v, idist, dx, dy, HCalib, PRE_RTll,
                             PRE_tTll, drescale, ps_t, Ku, Kv, KliP,
                             new_idist)) {
      tmp_res.new_res_state = ResState::OOB;
      return tmp_res.res_energy;
    }

    /// in the new frame
    Vec3f hit_color = (GetInterpolatedElement33(dIl, Ku, Kv, wG[0]));

    if (!std::isfinite((float)hit_color[0])) {
      tmp_res.new_res_state = ResState::OOB;
      return tmp_res.res_energy;
    }
    /// I_i = a(I_j - b)
    float residual = hit_color[0] - (affLL[0] * color[idx] + affLL[1]);

    float hw =
        fabsf(residual) < settingHuberTh ? 1 : settingHuberTh / fabsf(residual);
    energy_left +=
        weights[idx] * weights[idx] * hw * residual * residual * (2 - hw);

    /// depth derivatives, \f$ d(I_T) / d(\rho_H) = d(I_T) / d(u_T,v_T) *
    /// d(u_T,v_T) / d(\rho_H)\f$
    float dx_interp = hit_color[1] * HCalib.fxl();
    float dy_interp = hit_color[2] * HCalib.fyl();
    float d_idist = DeriveIdist(PRE_tTll, ps_t, dx_interp, dy_interp, drescale,
                                HCalib.xil());

    hw *= weights[idx] * weights[idx];

    /// JTJ, JTr
    Hdd += (hw * d_idist) * d_idist;
    bd += (hw * residual) * d_idist;
  }

  if (energy_left > energy_th * outlier_th_slack) {
    energy_left = energy_th * outlier_th_slack;
    tmp_res.new_res_state = ResState::OUTLIER;
  } else {
    tmp_res.new_res_state = ResState::IN;
  }

  tmp_res.new_res_energy = energy_left;
  return energy_left;
}

ImmaturePointStatus ImmaturePoint::TraceOn(FrameHessian& frame,
                                           const Mat33f& host_to_frame_KRKi,
                                           const Mat33f& host_to_frame_R,
                                           const Vec3f& host_to_frame_t,
                                           const Vec2f& host_to_frame_aff,
                                           CalibHessian& HCalib) {
  if (last_trace_status == ImmaturePointStatus::IPS_OOB) {
    return last_trace_status;
  }

  float max_pix_search = (wG[0] + hG[0]) * settingMaxPixSearch;
  // ============== project min and max. return if one of them is OOB
  Vec3f xh = Vec3f(x, y, z);
  Vec3f pr = host_to_frame_R * xh;  // Eqn. (G.7)
  Vec3f ptp_min = pr + host_to_frame_t * idist_min;
  Vec3f ptp_min_normalized = ptp_min.normalized();

  Vec3f ptp_min_pix = SpaceToPlane(ptp_min, HCalib.fxl(), HCalib.fyl(),
                                   HCalib.cxl(), HCalib.cyl(), HCalib.xil());
  Vec3f ptp_max_pix;
  float u_min = ptp_min_pix.x();
  float v_min = ptp_min_pix.y();

  if (!(u_min > 4 && v_min > 4 && u_min < wG[0] - 5 && v_min < hG[0] - 5 &&
        ValidArea(maskG[0], u_min, v_min))) {
    last_trace_uv = Vec2f(-1, -1);
    last_trace_pixel_interval = 0;
    return last_trace_status = ImmaturePointStatus::IPS_OOB;
  }

  float dist;
  float u_max;
  float v_max;
  Vec3f ptp_max;
  if (std::isfinite(idist_max)) {
    ptp_max = pr + host_to_frame_t * idist_max;
    ptp_max_pix = SpaceToPlane(ptp_max, HCalib.fxl(), HCalib.fyl(),
                               HCalib.cxl(), HCalib.cyl(), HCalib.xil());
    u_max = ptp_max_pix.x();
    v_max = ptp_max_pix.y();

    if (!(u_max > 4 && v_max > 4 && u_max < wG[0] - 5 && v_max < hG[0] - 5 &&
          ValidArea(maskG[0], u_max, v_max))) {
      last_trace_uv = Vec2f(-1, -1);
      last_trace_pixel_interval = 0;
      return last_trace_status = ImmaturePointStatus::IPS_OOB;
    }

    // TODO: use angular instead?

    // ============== check their distance. everything below 2px is OK (->
    // skip). ===================
    dist =
        (u_min - u_max) * (u_min - u_max) + (v_min - v_max) * (v_min - v_max);
    dist = sqrtf(dist);
    if (dist < settingTraceSlackInterval) {
      last_trace_uv = Vec2f(u_max + u_min, v_max + v_min) * 0.5;
      last_trace_pixel_interval = dist;
      return last_trace_status = ImmaturePointStatus::IPS_SKIPPED;
    }
    assert(dist > 0);
  } else {  // max = NAN, first traceOn
    dist = max_pix_search;

    // project to arbitrary depth to get direction.
    ptp_max = pr + host_to_frame_t * 0.01;
    ptp_max_pix = SpaceToPlane(ptp_max, HCalib.fxl(), HCalib.fyl(),
                               HCalib.cxl(), HCalib.cyl(), HCalib.xil());

    u_max = ptp_max_pix.x();
    v_max = ptp_max_pix.y();

    // direction.
    float dx = u_max - u_min;
    float dy = v_max - v_min;
    float d = 1.0f / sqrtf(dx * dx + dy * dy);

    // set to [settingMaxPixSearch].
    u_max = u_min + dist * dx * d;
    v_max = v_min + dist * dy * d;

    float idistMax_approx = 100;
    ApproxPixelCoordinate(xh, u_max, v_max, host_to_frame_R, host_to_frame_t,
                          HCalib, idistMax_approx, ptp_max_pix);
    u_max = ptp_max_pix.x();
    v_max = ptp_max_pix.y();
    ptp_max = LiftToSphere(u_max, v_max, HCalib.fxli(), HCalib.fyli(),
                           HCalib.cxli(), HCalib.cyli(), HCalib.xil());
    dist =
        (u_min - u_max) * (u_min - u_max) + (v_min - v_max) * (v_min - v_max);
    dist = sqrtf(dist);

    // may still be out!
    if (!(u_max > 4 && v_max > 4 && u_max < wG[0] - 5 && v_max < hG[0] - 5 &&
          ValidArea(maskG[0], u_max, v_max))) {
      last_trace_uv = Vec2f(-1, -1);
      last_trace_pixel_interval = 0;
      return last_trace_status = ImmaturePointStatus::IPS_OOB;
    }
    assert(dist > 0);
  }

  ptp_max =
      LiftToSphere(ptp_max_pix.x(), ptp_max_pix.y(), HCalib.fxli(),
                   HCalib.fyli(), HCalib.cxli(), HCalib.cyli(), HCalib.xil());
  Vec3f ptp_max_normalized = ptp_max.normalized();

  // set OOB if scale change too big.
  if (!(idist_min < 0 || (ptp_min.norm() > 0.75 && ptp_min.norm() < 1.5))) {
    last_trace_uv = Vec2f(-1, -1);
    last_trace_pixel_interval = 0;
    return last_trace_status = ImmaturePointStatus::IPS_OOB;
  }

  // ============== compute error-bounds on result in pixel. if the new interval
  // is not at least 1/2 of the old, SKIP ===================
  Vec3f ptp_min_add = pr + host_to_frame_t * (idist_min + 0.01);

  Vec3f pix_min_add = SpaceToPlane(ptp_min_add, HCalib.fxl(), HCalib.fyl(),
                                   HCalib.cxl(), HCalib.cyl(), HCalib.xil());
  float u_min_add = pix_min_add.x();
  float v_min_add = pix_min_add.y();

  //  float dx = settingTraceStepsize * (u_max - u_min);  /// will be normalized
  //  later float dy = settingTraceStepsize * (v_max - v_min);
  float dx =
      settingTraceStepsize * (u_min_add - u_min);  /// will be normalized later
  float dy = settingTraceStepsize * (v_min_add - v_min);
  float dxdy_norm = sqrt(dx * dx + dy * dy);
  dx /= dxdy_norm;
  dy /= dxdy_norm;

  float a = (Vec2f(dx, dy).transpose() * gradH * Vec2f(dx, dy));
  float b = (Vec2f(dy, -dx).transpose() * gradH * Vec2f(dy, -dx));
  float error_in_pixel = 0.2f + 0.2f * (a + b) / a;

  // NOTE: ??? error_in_pixel, a little different from "Semi-Dense Visual
  // Odometry for a Monocular Camera"
  if (error_in_pixel * settingTraceMinImprovementFactor > dist &&
      std::isfinite(idist_max)) {
    last_trace_uv = Vec2f(u_max + u_min, v_max + v_min) * 0.5;
    last_trace_pixel_interval = dist;
    return last_trace_status = ImmaturePointStatus::IPS_BADCONDITION;
  }

  if (error_in_pixel > 10) error_in_pixel = 10;

  // ============== do the discrete search ===================
  //  dx /= dist; /// normalize
  //  dy /= dist;

  if (dist > max_pix_search) {
    u_max = u_min + max_pix_search * dx;  // WARNING: approximation
    v_max = v_min + max_pix_search * dy;
    {
      float idist_max_approx;
      ApproxPixelCoordinate(xh, u_max, v_max, host_to_frame_R, host_to_frame_t,
                            HCalib, idist_max_approx, ptp_max_pix);
      u_max = ptp_max_pix.x();
      v_max = ptp_max_pix.y();
      ptp_max_normalized =
          LiftToSphere(u_max, v_max, HCalib.fxli(), HCalib.fyli(),
                       HCalib.cxli(), HCalib.cyli(), HCalib.xil());
      dist =
          (u_min - u_max) * (u_min - u_max) + (v_min - v_max) * (v_min - v_max);
      dist = sqrtf(dist);
    }
  }

  int num_steps = 1.9999f + dist / settingTraceStepsize;
  Mat22f Rplane = host_to_frame_KRKi.topLeftCorner<2, 2>();  // approximation

  float alpha = 1.0 / num_steps;

  float rand_shift = u_min * 1000 - floorf(u_min * 1000);
  Vec3f ptXYZ_alpha = alpha * (ptp_max_normalized - ptp_min_normalized);
  Vec3f ptXYZ = ptp_min_normalized;
  Vec3f ptXYZ_pix = SpaceToPlane(ptXYZ, HCalib.fxl(), HCalib.fyl(),
                                 HCalib.cxl(), HCalib.cyl(), HCalib.xil());
  float ptx = ptXYZ_pix.x();
  float pty = ptXYZ_pix.y();

  /// since KRKi * [du dv 0], only topleft will be used
  Vec2f rotatet_pattern[MAX_RES_PER_POINT];
  for (int idx = 0; idx < patternNum; idx++) {
    rotatet_pattern[idx] = Rplane * Vec2f(patternP[idx][0], patternP[idx][1]);
  }

  if (!std::isfinite(dx) || !std::isfinite(dy)) {
    last_trace_pixel_interval = 0;
    last_trace_uv = Vec2f(-1, -1);
    return last_trace_status = ImmaturePointStatus::IPS_OOB;
  }

  float errors[100];
  float best_u = 0, best_v = 0, best_energy = 1e10;
  int best_idx = -1;
  if (num_steps >= 100) num_steps = 99;

  /// numStep of 1D search * 8 pattern, find the best match (residual the
  /// smallest)
  for (int i = 0; i < num_steps; i++) {
    float energy = 0;
    for (int idx = 0; idx < patternNum; idx++) {
      float hit_color = NAN;
      if ((float)(ptx + rotatet_pattern[idx][0]) >= 0 &&
          (float)(pty + rotatet_pattern[idx][1]) >= 0 &&
          (float)(ptx + rotatet_pattern[idx][0]) < wG[0] &&
          (float)(pty + rotatet_pattern[idx][1]) < hG[0] &&
          ValidArea(maskG[0], ptx + rotatet_pattern[idx][0],
                    pty + rotatet_pattern[idx][1])) {
        hit_color = GetInterpolatedElement31(
            frame.dI, (float)(ptx + rotatet_pattern[idx][0]),
            (float)(pty + rotatet_pattern[idx][1]), wG[0]);
      }

      if (!std::isfinite(hit_color)) {
        energy += 1e5;
        continue;
      }
      float residual = hit_color - (float)(host_to_frame_aff[0] * color[idx] +
                                           host_to_frame_aff[1]);
      float hw =
          fabs(residual) < settingHuberTh ? 1 : settingHuberTh / fabs(residual);
      energy += hw * residual * residual * (2 - hw);
    }

    errors[i] = energy;
    if (energy < best_energy) {
      best_u = ptx;
      best_v = pty;
      best_energy = energy;
      best_idx = i;
    }

    ptXYZ = ptp_min_normalized + ptXYZ_alpha * (i + 1);
    ptXYZ.normalize();
    ptXYZ_pix = SpaceToPlane(ptXYZ, HCalib.fxl(), HCalib.fyl(), HCalib.cxl(),
                             HCalib.cyl(), HCalib.xil());
    ptx = ptXYZ_pix.x();
    pty = ptXYZ_pix.y();
  }

  /// find best score outside a +-2px radius.
  float second_best = 1e10;
  for (int i = 0; i < num_steps; i++) {
    if ((i < best_idx - settingMinTraceTestRadius ||
         i > best_idx + settingMinTraceTestRadius) &&
        errors[i] < second_best) {
      second_best = errors[i];
    }
  }
  /// like sth of non-max suppression, to filter out some similiar points in
  /// activatePoints
  float new_quality = second_best / best_energy;
  if (new_quality < quality || num_steps > 10) quality = new_quality;

  /// ============== do GN optimization =================== // max 3
  {
    Vec3f xt = LiftToSphere(best_u, best_v, HCalib.fxli(), HCalib.fyli(),
                            HCalib.cxli(), HCalib.cyli(), HCalib.xil());
    float idist_tmp =
        IdistTriangulation(xh, xt, host_to_frame_R, host_to_frame_t);
    float idistBest = std::isnan(idist_tmp) ? idist_min : idist_tmp;
    Vec3f ptp_best_add = pr + host_to_frame_t * (idistBest + 0.01);

    Vec3f pix_best_add = SpaceToPlane(ptp_best_add, HCalib.fxl(), HCalib.fyl(),
                                      HCalib.cxl(), HCalib.cyl(), HCalib.xil());
    float u_best_add = pix_best_add.x();
    float v_best_add = pix_best_add.y();
    /// will be normalized later
    dx = settingTraceStepsize * (u_best_add - best_u);
    dy = settingTraceStepsize * (v_best_add - best_v);
    dxdy_norm = sqrt(dx * dx + dy * dy);
    dx /= dxdy_norm;
    dy /= dxdy_norm;
  }
  float u_bak = best_u, v_bak = best_v, gnstepsize = 1, step_back = 0;
  if (settingTraceGNIterations > 0) best_energy = 1e5;
  int gn_steps_good = 0, gn_steps_bad = 0;
  for (int it = 0; it < settingTraceGNIterations; it++) {
    float H = 1, b = 0, energy = 0;
    for (int idx = 0; idx < patternNum; idx++) {
      Vec3f hit_color(NAN, NAN, NAN);
      if ((best_u + rotatet_pattern[idx][0]) >= 0 &&
          (best_u + rotatet_pattern[idx][0]) < wG[0] &&
          (best_v + rotatet_pattern[idx][1]) >= 0 &&
          (best_v + rotatet_pattern[idx][1]) < hG[0] &&
          ValidArea(maskG[0], best_u + rotatet_pattern[idx][0],
                    best_v + rotatet_pattern[idx][1])) {
        hit_color = GetInterpolatedElement33(
            frame.dI, (float)(best_u + rotatet_pattern[idx][0]),
            (float)(best_v + rotatet_pattern[idx][1]), wG[0]);
      }

      if (!std::isfinite((float)hit_color[0])) {
        energy += 1e5;
        continue;
      }
      float residual = hit_color[0] - (host_to_frame_aff[0] * color[idx] +
                                       host_to_frame_aff[1]);
      /// J^res_dist, [dx dy] = d_pixel/d_dist
      float dResdDist = dx * hit_color[1] + dy * hit_color[2];
      float hw =
          fabs(residual) < settingHuberTh ? 1 : settingHuberTh / fabs(residual);

      H += hw * dResdDist * dResdDist;
      b += hw * residual * dResdDist;
      energy +=
          weights[idx] * weights[idx] * hw * residual * residual * (2 - hw);
    }

    if (energy > best_energy) {
      gn_steps_bad++;

      // do a smaller step from old point.
      step_back *= 0.5;
      best_u = u_bak + step_back * dx;
      best_v = v_bak + step_back * dy;

      float idist_best;
      Vec3f best_uv;
      ApproxPixelCoordinate(xh, best_u, best_v, host_to_frame_R,
                            host_to_frame_t, HCalib, idist_best, best_uv);

      best_u = best_uv.x();
      best_v = best_uv.y();
    } else {
      gn_steps_good++;

      float step = -gnstepsize * b / H;
      if (step < -0.5)
        step = -0.5;
      else if (step > 0.5)
        step = 0.5;

      if (!std::isfinite(step)) step = 0;

      u_bak = best_u;
      v_bak = best_v;
      step_back = step;

      best_u += step * dx;
      best_v += step * dy;
      best_energy = energy;

      float idist_best;
      Vec3f best_uv;
      ApproxPixelCoordinate(xh, best_u, best_v, host_to_frame_R,
                            host_to_frame_t, HCalib, idist_best, best_uv);

      best_u = best_uv.x();
      best_v = best_uv.y();
    }

    if (fabsf(step_back) < settingTraceGNThreshold) break;
  }

  // ============== detect energy-based outlier. ===================
  // float absGrad0 = getInterpolatedElement(frame.absSquaredGrad[0],best_u,
  // best_v, wG[0]); float absGrad1 =
  // getInterpolatedElement(frame.absSquaredGrad[1],best_u*0.5-0.25,
  // best_v*0.5-0.25, wG[1]); float absGrad2 =
  // getInterpolatedElement(frame.absSquaredGrad[2],best_u*0.25-0.375,
  // best_v*0.25-0.375, wG[2]);
  /// if bestEnergy too large
  if (!(best_energy < energy_th * settingTraceExtraSlackOnTH))
  // || (absGrad0*areaGradientSlackFactor < host->frameGradTH
  // && absGrad1*areaGradientSlackFactor < host->frameGradTH*0.75f
  // && absGrad2*areaGradientSlackFactor < host->frameGradTH*0.50f))
  {
    last_trace_pixel_interval = 0;
    last_trace_uv = Vec2f(-1, -1);
    /// NOTE: for repeated and occlusion ???
    if (last_trace_status == ImmaturePointStatus::IPS_OUTLIER)
      return last_trace_status = ImmaturePointStatus::IPS_OOB;
    else
      return last_trace_status = ImmaturePointStatus::IPS_OUTLIER;
  }

  /// ============== set new interval ===================
  Vec3f vh(x, y, z);
  Vec3f vt;
  if (dx * dx > dy * dy) {
    float idist_tmp;
    vt =
        LiftToSphere((best_u - error_in_pixel * dx),
                     (best_v - error_in_pixel * dy), HCalib.fxli(),
                     HCalib.fyli(), HCalib.cxli(), HCalib.cyli(), HCalib.xil());
    idist_tmp = IdistTriangulation(vh, vt, host_to_frame_R, host_to_frame_t);
    idist_min = std::isnan(idist_tmp) ? idist_min : idist_tmp;
    vt =
        LiftToSphere((best_u + error_in_pixel * dx),
                     (best_v + error_in_pixel * dy), HCalib.fxli(),
                     HCalib.fyli(), HCalib.cxli(), HCalib.cyli(), HCalib.xil());
    idist_tmp = IdistTriangulation(vh, vt, host_to_frame_R, host_to_frame_t);
    idist_max = std::isnan(idist_tmp) ? idist_max : idist_tmp;
  } else {
    float idist_tmp;
    vt =
        LiftToSphere((best_u - error_in_pixel * dx),
                     (best_v - error_in_pixel * dy), HCalib.fxli(),
                     HCalib.fyli(), HCalib.cxli(), HCalib.cyli(), HCalib.xil());
    idist_tmp = IdistTriangulation(vh, vt, host_to_frame_R, host_to_frame_t);
    idist_min = std::isnan(idist_tmp) ? idist_min : idist_tmp;
    vt =
        LiftToSphere((best_u + error_in_pixel * dx),
                     (best_v + error_in_pixel * dy), HCalib.fxli(),
                     HCalib.fyli(), HCalib.cxli(), HCalib.cyli(), HCalib.xil());
    idist_tmp = IdistTriangulation(vh, vt, host_to_frame_R, host_to_frame_t);
    idist_max = std::isnan(idist_tmp) ? idist_max : idist_tmp;
  }
  if (idist_min > idist_max) std::swap<float>(idist_min, idist_max);

  // DLOG(INFO) << "minmax: " << idist_min << " " << idist_max;
  if (!std::isfinite(idist_min) || !std::isfinite(idist_max) ||
      (idist_max < 0)) {
    last_trace_pixel_interval = 0;
    last_trace_uv = Vec2f(-1, -1);
    return last_trace_status = ImmaturePointStatus::IPS_OUTLIER;
  }

  last_trace_pixel_interval = 2 * error_in_pixel;
  last_trace_uv = Vec2f(best_u, best_v);
  return last_trace_status = ImmaturePointStatus::IPS_GOOD;
}

}  // namespace dsl
