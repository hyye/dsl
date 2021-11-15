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
// Created by hyye on 11/7/19.
//

#include "full_system/coarse_tracker.h"

namespace dsl {

CoarseTracker::CoarseTracker(int ww, int hh) : last_ref_aff_light(0, 0) {
  for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
    int wl = ww >> lvl;
    int hl = hh >> lvl;

    idist[lvl] = std::move(std::vector<float>(wl * hl));
    weight_sums[lvl] = std::move(std::vector<float>(wl * hl));
    weight_sums_bak[lvl] = std::move(std::vector<float>(wl * hl));

    pc_u[lvl] = std::move(std::vector<float>(wl * hl));
    pc_v[lvl] = std::move(std::vector<float>(wl * hl));
    pc_x[lvl] = std::move(std::vector<float>(wl * hl));
    pc_y[lvl] = std::move(std::vector<float>(wl * hl));
    pc_z[lvl] = std::move(std::vector<float>(wl * hl));
    pc_idist[lvl] = std::move(std::vector<float>(wl * hl));
    pc_color[lvl] = std::move(std::vector<float>(wl * hl));
  }

  // warped buffers
  buf_warped_idist = std::move(std::vector<float>(ww * hh));
  buf_warped_u = std::move(std::vector<float>(ww * hh));
  buf_warped_v = std::move(std::vector<float>(ww * hh));
  buf_warped_x = std::move(std::vector<float>(ww * hh));
  buf_warped_y = std::move(std::vector<float>(ww * hh));
  buf_warped_z = std::move(std::vector<float>(ww * hh));
  buf_warped_idepth_xi = std::move(std::vector<float>(ww * hh));
  buf_warped_dx = std::move(std::vector<float>(ww * hh));
  buf_warped_dy = std::move(std::vector<float>(ww * hh));
  buf_warped_residual = std::move(std::vector<float>(ww * hh));
  buf_warped_weight = std::move(std::vector<float>(ww * hh));
  buf_warped_ref_color = std::move(std::vector<float>(ww * hh));

  new_frame = nullptr;
  last_ref_fh = nullptr;
  w[0] = h[0] = 0;
  ref_frame_id = -1;
}

CoarseTracker::~CoarseTracker() {}

void CoarseTracker::MakeK(CalibHessian &HCalib) {
  w[0] = wG[0];
  h[0] = hG[0];

  fx[0] = HCalib.fxl();
  fy[0] = HCalib.fyl();
  cx[0] = HCalib.cxl();
  cy[0] = HCalib.cyl();
  xi = HCalib.xil();

  for (int level = 1; level < pyrLevelsUsed; ++level) {
    w[level] = w[0] >> level;
    h[level] = h[0] >> level;
    fx[level] = fx[level - 1] * 0.5;
    fy[level] = fy[level - 1] * 0.5;
    cx[level] = (cx[0] + 0.5) / ((int)1 << level) - 0.5;
    cy[level] = (cy[0] + 0.5) / ((int)1 << level) - 0.5;
  }

  for (int level = 0; level < pyrLevelsUsed; ++level) {
    K[level] << fx[level], 0.0, cx[level], 0.0, fy[level], cy[level], 0.0, 0.0,
        1.0;
    Ki[level] = K[level].inverse();
    fxi[level] = Ki[level](0, 0);
    fyi[level] = Ki[level](1, 1);
    cxi[level] = Ki[level](0, 2);
    cyi[level] = Ki[level](1, 2);
  }
}

/**
 * MakeCoarseDistL0, after optimization
 * @param frame_hessians from optimized
 */
void CoarseTracker::MakeCoarseDistL0(
    std::vector<std::unique_ptr<FrameHessian>> &frame_hessians) {
  std::fill(idist[0].begin(), idist[0].end(), 0);
  std::fill(weight_sums[0].begin(), weight_sums[0].end(), 0);

  if (frame_hessians.size() == 1) {
    for (std::unique_ptr<FrameHessian> &fh : frame_hessians) {
      for (std::unique_ptr<PointHessian> &ph : fh->point_hessians) {
        int u = ph->u;
        int v = ph->v;
        float new_idist = ph->idist;
        float weight = 100;
        idist[0][u + w[0] * v] += new_idist * weight;
        weight_sums[0][u + w[0] * v] += weight;
      }
    }
  } else {
    for (std::unique_ptr<FrameHessian> &fh : frame_hessians) {
      for (std::unique_ptr<PointHessian> &ph : fh->point_hessians) {
        /// target is the new frame
        if (ph->last_residuals[0].first != 0 &&
            ph->last_residuals[0].second == ResState::IN) {
          PointFrameResidual *r = ph->last_residuals[0].first;
          assert(r->ef_residual->IsActive() && r->target == last_ref_fh);
          int u = r->center_projected_to[0] + 0.5f;
          int v = r->center_projected_to[1] + 0.5f;
          float new_idist = r->center_projected_to[2];
          float weight = sqrtf(1e-3 / (ph->ef_point->HdiF + 1e-12));
          // FIXME: the HdiF!!!

          /// for the converged points
          if (ph->convereged_ph_idist) {
            weight = 100;
          }

          idist[0][u + w[0] * v] += new_idist * weight;
          weight_sums[0][u + w[0] * v] += weight;
        }
      }
    }
  }

  // auto NonZ = [](const std::vector<float> &weights) {
  //   int cnt = 0;
  //   for (int i = 0; i < weights.size(); ++i) {
  //     if (weights[i] > 0) cnt++;
  //   }
  //   return cnt;
  // };

  for (int lvl = 1; lvl < pyrLevelsUsed; lvl++) {
    int lvlm1 = lvl - 1;
    int wl = w[lvl], hl = h[lvl], wlm1 = w[lvlm1];

    std::vector<float> &idist_l = idist[lvl];
    std::vector<float> &weightSums_l = weight_sums[lvl];

    std::vector<float> &idist_lm = idist[lvlm1];
    std::vector<float> &weightSums_lm = weight_sums[lvlm1];

    for (int y = 0; y < hl; y++)
      for (int x = 0; x < wl; x++) {
        int bidx = 2 * x + 2 * y * wlm1;
        idist_l[x + y * wl] = idist_lm[bidx] + idist_lm[bidx + 1] +
                              idist_lm[bidx + wlm1] + idist_lm[bidx + wlm1 + 1];

        weightSums_l[x + y * wl] =
            weightSums_lm[bidx] + weightSums_lm[bidx + 1] +
            weightSums_lm[bidx + wlm1] + weightSums_lm[bidx + wlm1 + 1];
      }

    // LOG(INFO) << "lvl: " << lvl << " w1: " << NonZ(weight_sums[lvl]);
  }

  // dilate idist by 1.
  for (int lvl = 0; lvl < 2; lvl++) {
    int numIts = 1;

    for (int it = 0; it < numIts; it++) {
      int wh = w[lvl] * h[lvl] - w[lvl];
      int wl = w[lvl];
      std::vector<float> &weight_sumsl = weight_sums[lvl];
      std::vector<float> &weight_sumsl_bak = weight_sums_bak[lvl];
      weight_sumsl_bak = weight_sumsl;
      std::vector<float> &idistl = idist[lvl];  // dotnt need to make a temp
                                                // copy of depth, since I only
      // read values with weight_sumsl>0, and write ones with weight_sumsl<=0.
      for (int i = w[lvl]; i < wh; i++) {
        if (weight_sumsl_bak[i] <= 0) {
          float sum = 0, num = 0, numn = 0;
          if (weight_sumsl_bak[i + 1 + wl] > 0) {
            sum += idistl[i + 1 + wl];
            num += weight_sumsl_bak[i + 1 + wl];
            numn++;
          }
          if (weight_sumsl_bak[i - 1 - wl] > 0) {
            sum += idistl[i - 1 - wl];
            num += weight_sumsl_bak[i - 1 - wl];
            numn++;
          }
          if (weight_sumsl_bak[i + wl - 1] > 0) {
            sum += idistl[i + wl - 1];
            num += weight_sumsl_bak[i + wl - 1];
            numn++;
          }
          if (weight_sumsl_bak[i - wl + 1] > 0) {
            sum += idistl[i - wl + 1];
            num += weight_sumsl_bak[i - wl + 1];
            numn++;
          }
          if (numn > 0) {
            idistl[i] = sum / numn;
            weight_sumsl[i] = num / numn;
          }
        }
      }
    }

    // LOG(INFO) << "lvl: " << lvl << " w2: " << NonZ(weight_sums[lvl]);
  }

  // dilate idist by 1 (2 on lower levels).
  for (int lvl = 2; lvl < pyrLevelsUsed; lvl++) {
    int wh = w[lvl] * h[lvl] - w[lvl];
    int wl = w[lvl];
    std::vector<float> &weight_sumsl = weight_sums[lvl];
    std::vector<float> &weight_sumsl_bak = weight_sums_bak[lvl];
    weight_sumsl_bak = weight_sumsl;
    std::vector<float> &idistl = idist[lvl];  // don't need to make a temp copy
                                              // of depth, since I only
    // read values with weight_sumsl>0, and write ones with weight_sumsl<=0.
    for (int i = w[lvl]; i < wh; i++) {
      if (weight_sumsl_bak[i] <= 0) {
        float sum = 0, num = 0, numn = 0;
        if (weight_sumsl_bak[i + 1] > 0) {
          sum += idistl[i + 1];
          num += weight_sumsl_bak[i + 1];
          numn++;
        }
        if (weight_sumsl_bak[i - 1] > 0) {
          sum += idistl[i - 1];
          num += weight_sumsl_bak[i - 1];
          numn++;
        }
        if (weight_sumsl_bak[i + wl] > 0) {
          sum += idistl[i + wl];
          num += weight_sumsl_bak[i + wl];
          numn++;
        }
        if (weight_sumsl_bak[i - wl] > 0) {
          sum += idistl[i - wl];
          num += weight_sumsl_bak[i - wl];
          numn++;
        }
        if (numn > 0) {
          idistl[i] = sum / numn;
          weight_sumsl[i] = num / numn;
        }
      }
    }

    // LOG(INFO) << "lvl: " << lvl << " w3: " << NonZ(weight_sums[lvl]);
  }

  // normalize idepths and weights.
  for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
    std::vector<float> &weight_sumsl = weight_sums[lvl];
    std::vector<float> &idistl = idist[lvl];
    std::vector<Vec3f> &dI_refl = last_ref_fh->dIp[lvl];

    int wl = w[lvl], hl = h[lvl];

    int lpc_n = 0;
    std::vector<float> &lpc_u = pc_u[lvl];
    std::vector<float> &lpc_v = pc_v[lvl];
    std::vector<float> &lpc_x = pc_x[lvl];
    std::vector<float> &lpc_y = pc_y[lvl];
    std::vector<float> &lpc_z = pc_z[lvl];
    std::vector<float> &lpc_idist = pc_idist[lvl];
    std::vector<float> &lpc_color = pc_color[lvl];

    for (int y = 2; y < hl - 2; y++)
      for (int x = 2; x < wl - 2; x++) {
        int i = x + y * wl;

        if (weight_sumsl[i] > 0) {
          idistl[i] /= weight_sumsl[i];
          lpc_u[lpc_n] = x;
          lpc_v[lpc_n] = y;

          Vec3f pcs = LiftToSphere((float)x, (float)y, fxi[lvl], fyi[lvl],
                                   cxi[lvl], cyi[lvl], xi);
          lpc_x[lpc_n] = pcs.x();
          lpc_y[lpc_n] = pcs.y();
          lpc_z[lpc_n] = pcs.z();

          lpc_idist[lpc_n] = idistl[i];
          lpc_color[lpc_n] = dI_refl[i][0];

          if (!std::isfinite(lpc_color[lpc_n]) || !(idistl[i] > 0)) {
            idistl[i] = -1;
            continue;  // just skip if something is wrong.
          }
          lpc_n++;
        } else
          idistl[i] = -1;

        weight_sumsl[i] = 1;
      }

    pc_n[lvl] = lpc_n;

    // LOG(INFO) << "lvl: " << lvl << " w4: " << NonZ(weight_sums[lvl]);
    // LOG(INFO) << "lvl: " << lvl << " lpc_n: " << lpc_n;
  }
}

void CoarseTracker::SetCoarseTrackingRef(
    std::vector<std::unique_ptr<FrameHessian>> &frame_hessians) {
  assert(frame_hessians.size() > 0);
  last_ref_fh = frame_hessians.back().get();
  MakeCoarseDistL0(frame_hessians);

  ref_frame_id = last_ref_fh->shell->id;
  last_ref_aff_light = last_ref_fh->GetAffLight();

  first_coarse_rmse = -1;

  /*
  for (int lvl = 0; lvl < pyrLevelsUsed; ++lvl) {
    LOG(INFO) << hG[lvl] << " " << wG[lvl] << " " << pyrLevelsUsed;
    cv::Mat idist_img = cv::Mat(hG[lvl], wG[lvl], CV_32FC1, idist[lvl].data());
    cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
    cv::imshow("Display window", idist_img);

    cv::waitKey(0);
  }
  */
}

bool CoarseTracker::TrackNewestCoarse(
    FrameHessian &new_fh, SE3 &last_to_new_out, AffLight &aff_light_out,
    int coarsest_lvl, const Eigen::Matrix<double, 5, 1> &min_res_for_abort) {
  assert(coarsest_lvl < 5 && coarsest_lvl < pyrLevelsUsed);
  last_residuals.setConstant(NAN);
  last_flow_indicators.setConstant(1000);
  new_frame = &new_fh;

  int max_iterations[] = {10, 20, 50, 50, 50};
  float lambda_extrapolation_limit = 0.001;

  SE3 ref_to_new_current = last_to_new_out;
  AffLight aff_light_current = aff_light_out;

  bool debug_print = false;
  bool have_repeated = false;

  for (int lvl = coarsest_lvl; lvl >= 0; lvl--) {
    Mat88 H;
    Vec8 b;
    float level_cutoff_repeat = 1;
    Vec6 res_old = CalcRes(lvl, ref_to_new_current, aff_light_current,
                           settingCoarseCutoffTh * level_cutoff_repeat);
    while (res_old[5] > 0.6 && level_cutoff_repeat < 50) {
      level_cutoff_repeat *= 2;
      res_old = CalcRes(lvl, ref_to_new_current, aff_light_current,
                        settingCoarseCutoffTh * level_cutoff_repeat);

      if (!settingDebugoutRunquiet) {
        char buff[100];
        snprintf(buff, sizeof(buff), "INCREASING cutoff to %f (ratio is %f)!\n",
                 settingCoarseCutoffTh * level_cutoff_repeat, res_old[5]);
        std::string buff_as_str = buff;
        LOG(INFO) << buff_as_str;
      }
    }

    CalcGSSSE(lvl, H, b, ref_to_new_current, aff_light_current);

    float lambda = 0.01;

    if (debug_print) {
      Vec2f relAff = AffLight::FromToVecExposure(
                         last_ref_fh->exposure, new_frame->exposure,
                         last_ref_aff_light, aff_light_current)
                         .cast<float>();
      char buff[100];
      snprintf(
          buff, sizeof(buff),
          "lvl%d, it %d (l=%f / %f) %s: %.3f->%.3f (%d -> %d) (|inc| = %f)! \t",
          lvl, -1, lambda, 1.0f, "INITIA", 0.0f, res_old[0] / res_old[1], 0,
          (int)res_old[1], 0.0f);
      std::string buff_as_str = buff;
      LOG(INFO) << buff_as_str;
      LOG(INFO) << ref_to_new_current.log().transpose() << " AFF "
                << aff_light_current.Vec().transpose() << " (rel "
                << relAff.transpose() << ")";
    }

    for (int iteration = 0; iteration < max_iterations[lvl]; iteration++) {
      Mat88 Hl = H;
      for (int i = 0; i < 8; i++) Hl(i, i) *= (1 + lambda);
      Vec8 inc = Hl.ldlt().solve(-b);

      if (settingAffineOptModeA < 0 && settingAffineOptModeB < 0)  // fix a, b
      {
        inc.head<6>() = Hl.topLeftCorner<6, 6>().ldlt().solve(-b.head<6>());
        inc.tail<2>().setZero();
      }
      if (!(settingAffineOptModeA < 0) && settingAffineOptModeB < 0)  // fix b
      {
        inc.head<7>() = Hl.topLeftCorner<7, 7>().ldlt().solve(-b.head<7>());
        inc.tail<1>().setZero();
      }
      if (settingAffineOptModeA < 0 && !(settingAffineOptModeB < 0))  // fix a
      {
        Mat88 Hl_stitch = Hl;
        Vec8 b_stitch = b;
        Hl_stitch.col(6) = Hl_stitch.col(7);
        Hl_stitch.row(6) = Hl_stitch.row(7);
        b_stitch[6] = b_stitch[7];
        Vec7 inc_stitch =
            Hl_stitch.topLeftCorner<7, 7>().ldlt().solve(-b_stitch.head<7>());
        inc.setZero();
        inc.head<6>() = inc_stitch.head<6>();
        inc[6] = 0;
        inc[7] = inc_stitch[6];
      }

      float extrap_fac = 1;
      if (lambda < lambda_extrapolation_limit) {
        extrap_fac = sqrt(sqrt(lambda_extrapolation_limit / lambda));
      }
      inc *= extrap_fac;

      Vec8 inc_scaled = inc;
      inc_scaled.segment<3>(0) *= SCALE_XI_TRANS;
      inc_scaled.segment<3>(3) *= SCALE_XI_ROT;
      inc_scaled.segment<1>(6) *= SCALE_A;
      inc_scaled.segment<1>(7) *= SCALE_B;

      if (!std::isfinite(inc_scaled.sum())) inc_scaled.setZero();

      SE3 ref_to_new_new =
          SE3::exp((Vec6)(inc_scaled.head<6>())) * ref_to_new_current;
      AffLight aff_light_new = aff_light_current;
      aff_light_new.a += inc_scaled[6];
      aff_light_new.b += inc_scaled[7];

      Vec6 res_new = CalcRes(lvl, ref_to_new_new, aff_light_new,
                             settingCoarseCutoffTh * level_cutoff_repeat);

      bool accept = (res_new[0] / res_new[1]) < (res_old[0] / res_old[1]);
      // LOG(INFO) << "lvl, iter: " << lvl << ", " << iteration;

      if (debug_print) {
        char buff[100];
        Vec2f ref_aff = AffLight::FromToVecExposure(
                            last_ref_fh->exposure, new_frame->exposure,
                            last_ref_aff_light, aff_light_new)
                            .cast<float>();
        snprintf(buff, sizeof(buff),
                 "lvl %d, it %d (l=%f / %f) %s: %.3f->%.3f (%d -> %d) (|inc| = "
                 "%f)! \t",
                 lvl, iteration, lambda, extrap_fac,
                 (accept ? "ACCEPT" : "REJECT"), res_old[0] / res_old[1],
                 res_new[0] / res_new[1], (int)res_old[1], (int)res_new[1],
                 inc.norm());
        std::string buff_as_str = buff;
        LOG(INFO) << buff_as_str;
        LOG(INFO) << ref_to_new_new.log().transpose() << " AFF "
                  << aff_light_new.Vec().transpose() << " (rel "
                  << ref_aff.transpose() << ")";
      }
      if (accept) {
        CalcGSSSE(lvl, H, b, ref_to_new_new, aff_light_new);
        res_old = res_new;
        aff_light_current = aff_light_new;
        ref_to_new_current = ref_to_new_new;
        lambda *= 0.5;
      } else {
        lambda *= 4;
        if (lambda < lambda_extrapolation_limit)
          lambda = lambda_extrapolation_limit;
      }

      if (!(inc.norm() > 1e-3)) {
        if (debug_print) LOG(INFO) << "inc too small, break!";
        break;
      }
    }

    // set last residual for that level, as well as flow indicators.
    last_residuals[lvl] = sqrtf((float)(res_old[0] / res_old[1]));
    last_flow_indicators = res_old.segment<3>(2);

    DLOG(WARNING) << ">>>>>>> lvl " << lvl << " res: " << last_residuals[lvl]
                  << " minRes: " << min_res_for_abort[lvl] << " pose: "
                  << (last_ref_fh->shell->cam_to_world *
                      ref_to_new_current.inverse())
                         .translation()
                         .transpose();

    if (last_residuals[lvl] > 1.5 * min_res_for_abort[lvl]) return false;

    if (level_cutoff_repeat > 1 && !have_repeated) {
      lvl++;
      have_repeated = true;
      LOG(INFO) << "REPEAT LEVEL!";
    }
  }

  // set!
  last_to_new_out = ref_to_new_current;
  aff_light_out = aff_light_current;

  if ((settingAffineOptModeA != 0 && (fabsf(aff_light_out.a) > 1.2)) ||
      (settingAffineOptModeB != 0 && (fabsf(aff_light_out.b) > 200)))
    return false;

  Vec2f ref_aff =
      AffLight::FromToVecExposure(last_ref_fh->exposure, new_frame->exposure,
                                  last_ref_aff_light, aff_light_out)
          .cast<float>();

  DLOG(WARNING) << "ref_aff: " << fabsf(logf((float)ref_aff[0])) << " "
                << fabsf((float)ref_aff[1]);

  if ((settingAffineOptModeA == 0 && (fabsf(logf((float)ref_aff[0])) > 1.5)) ||
      (settingAffineOptModeB == 0 && (fabsf((float)ref_aff[1]) > 200)))
    return false;

  if (settingAffineOptModeA < 0) aff_light_out.a = 0;
  if (settingAffineOptModeB < 0) aff_light_out.b = 0;

  return true;
}

Vec6 CoarseTracker::CalcRes(int lvl, const SE3 &ref_to_new, AffLight aff_light,
                            float cutoff_th) {
  float E = 0;
  int num_terms_in_E = 0;
  int num_terms_in_warped = 0;
  int num_saturated = 0;

  int wl = w[lvl];
  int hl = h[lvl];
  std::vector<Vec3f> &dI_newl = new_frame->dIp[lvl];
  float fxl = fx[lvl];
  float fyl = fy[lvl];
  float cxl = cx[lvl];
  float cyl = cy[lvl];
  float xil = xi;

  Mat33f R = ref_to_new.rotationMatrix().cast<float>();
  Vec3f t = (ref_to_new.translation()).cast<float>();
  Vec2f affLL =
      AffLight::FromToVecExposure(last_ref_fh->exposure, new_frame->exposure,
                                  last_ref_aff_light, aff_light)
          .cast<float>();

  float sum_sq_shift_T = 0;
  float sum_sq_shift_RT = 0;
  float sum_squared_shift_num = 0;

  // energy for r=settingCoarseCutoffTh.
  float max_energy =
      2 * settingHuberTh * cutoff_th - settingHuberTh * settingHuberTh;

  // WARNING: no debug_plot

  int nl = pc_n[lvl];
  std::vector<float> &lpc_u = pc_u[lvl];
  std::vector<float> &lpc_v = pc_v[lvl];
  std::vector<float> &lpc_x = pc_x[lvl];
  std::vector<float> &lpc_y = pc_y[lvl];
  std::vector<float> &lpc_z = pc_z[lvl];
  std::vector<float> &lpc_idist = pc_idist[lvl];
  std::vector<float> &lpc_color = pc_color[lvl];

  for (int i = 0; i < nl; i++) {
    float id = lpc_idist[i];
    float u = lpc_u[i];
    float v = lpc_v[i];
    float x = lpc_x[i];
    float y = lpc_y[i];
    float z = lpc_z[i];

    Vec3f pt = R * Vec3f(x, y, z) + t * id;
    float new_idist = id / pt.norm();
    Vec3f Kuv = SpaceToPlane(pt, fxl, fyl, cxl, cyl, xil);
    float Ku = Kuv.x(), Kv = Kuv.y();

    if (lvl == 0 && i % 32 == 0) {
      // translation only (positive)
      Vec3f ptT = Vec3f(x, y, z) + t * id;
      Vec3f KuvT = SpaceToPlane(ptT, fxl, fyl, cxl, cyl, xil);
      float KuT = KuvT.x(), KvT = KuvT.y();

      // translation only (negative)
      Vec3f ptT2 = Vec3f(x, y, z) - t * id;
      Vec3f KuvT2 = SpaceToPlane(ptT2, fxl, fyl, cxl, cyl, xil);
      float KuT2 = KuvT2.x(), KvT2 = KuvT2.y();

      // translation and rotation (negative)
      Vec3f pt3 = R * Vec3f(x, y, z) - t * id;
      Vec3f Kuv3 = SpaceToPlane(pt3, fxl, fyl, cxl, cyl, xil);
      float Ku3 = Kuv3.x(), Kv3 = Kuv3.y();

      // translation and rotation (positive)
      // already have it.

      sum_sq_shift_T += (KuT - u) * (KuT - u) + (KvT - v) * (KvT - v);
      sum_sq_shift_T += (KuT2 - u) * (KuT2 - u) + (KvT2 - v) * (KvT2 - v);
      sum_sq_shift_RT += (Ku - u) * (Ku - u) + (Kv - v) * (Kv - v);
      sum_sq_shift_RT += (Ku3 - u) * (Ku3 - u) + (Kv3 - v) * (Kv3 - v);
      sum_squared_shift_num += 2;
    }

    if (!(Ku > 2 && Kv > 2 && Ku < wl - 3 && Kv < hl - 3 && new_idist > 0 &&
          ValidArea(maskG[lvl], Ku, Kv)))
      continue;

    float ref_color = lpc_color[i];
    Vec3f hit_color = GetInterpolatedElement33(dI_newl.data(), Ku, Kv, wl);
    if (!std::isfinite((float)hit_color[0])) continue;
    float residual = hit_color[0] - (float)(affLL[0] * ref_color + affLL[1]);
    float hw =
        fabs(residual) < settingHuberTh ? 1 : settingHuberTh / fabs(residual);

    if (fabs(residual) > cutoff_th) {
      E += max_energy;
      num_terms_in_E++;
      num_saturated++;
    } else {
      E += hw * residual * residual * (2 - hw);
      num_terms_in_E++;

      buf_warped_idist[num_terms_in_warped] = new_idist;
      buf_warped_u[num_terms_in_warped] = Ku;
      buf_warped_v[num_terms_in_warped] = Kv;
      pt.normalize();  // as warped on the unit sphere
      buf_warped_x[num_terms_in_warped] = pt.x();
      buf_warped_y[num_terms_in_warped] = pt.y();
      buf_warped_z[num_terms_in_warped] = pt.z();
      buf_warped_idepth_xi[num_terms_in_warped] = 1 / (pt.z() + xi);
      buf_warped_dx[num_terms_in_warped] = hit_color[1];
      buf_warped_dy[num_terms_in_warped] = hit_color[2];
      buf_warped_residual[num_terms_in_warped] = residual;
      buf_warped_weight[num_terms_in_warped] = hw;
      buf_warped_ref_color[num_terms_in_warped] = lpc_color[i];
      num_terms_in_warped++;
    }
  }

  while (num_terms_in_warped % 4 != 0) {
    buf_warped_idist[num_terms_in_warped] = 0;
    buf_warped_u[num_terms_in_warped] = 0;
    buf_warped_v[num_terms_in_warped] = 0;
    buf_warped_x[num_terms_in_warped] = 0;
    buf_warped_y[num_terms_in_warped] = 0;
    buf_warped_z[num_terms_in_warped] = 0;
    buf_warped_idepth_xi[num_terms_in_warped] = 0;
    buf_warped_dx[num_terms_in_warped] = 0;
    buf_warped_dy[num_terms_in_warped] = 0;
    buf_warped_residual[num_terms_in_warped] = 0;
    buf_warped_weight[num_terms_in_warped] = 0;
    buf_warped_ref_color[num_terms_in_warped] = 0;
    num_terms_in_warped++;
  }
  buf_warped_n = num_terms_in_warped;

  Vec6 rs;
  rs[0] = E;
  rs[1] = num_terms_in_E;
  rs[2] = sum_sq_shift_T / (sum_squared_shift_num + 0.1);
  rs[3] = 0;
  rs[4] = sum_sq_shift_RT / (sum_squared_shift_num + 0.1);
  rs[5] = num_saturated / (float)num_terms_in_E;

  //  LOG(ERROR) << sum_sq_shift_T << " " << sum_sq_shift_RT;

  return rs;
}

void CoarseTracker::CalcGSSSE(int lvl, Mat88 &H_out, Vec8 &b_out,
                              const SE3 &ref_to_new, AffLight aff_light) {
  acc.Initialize();

  __m128 fxl = _mm_set1_ps(fx[lvl]);
  __m128 fyl = _mm_set1_ps(fy[lvl]);
  __m128 b0 = _mm_set1_ps(last_ref_aff_light.b);
  __m128 a = _mm_set1_ps((float)(AffLight::FromToVecExposure(
      last_ref_fh->exposure, new_frame->exposure, last_ref_aff_light,
      aff_light)[0]));

  __m128 one = _mm_set1_ps(1);
  __m128 minusOne = _mm_set1_ps(-1);
  __m128 zero = _mm_set1_ps(0);
  __m128 xi_m128 = _mm_set1_ps(xi);

  int n = buf_warped_n;
  assert(n % 4 == 0);
  for (int i = 0; i < n; i += 4) {
    __m128 xs = _mm_load_ps(buf_warped_x.data() + i);
    __m128 ys = _mm_load_ps(buf_warped_y.data() + i);
    __m128 zs = _mm_load_ps(buf_warped_z.data() + i);
    __m128 id = _mm_load_ps(buf_warped_idist.data() + i);
    __m128 idepth_xi = _mm_load_ps(buf_warped_idepth_xi.data() + i);

    __m128 xsys = _mm_mul_ps(xs, ys);
    __m128 yszs = _mm_mul_ps(ys, zs);
    __m128 xszs = _mm_mul_ps(xs, zs);

    __m128 fact_r = _mm_mul_ps(idepth_xi, idepth_xi);
    __m128 fact_t = _mm_mul_ps(id, fact_r);

    __m128 one_minus_xs2 = _mm_sub_ps(one, _mm_mul_ps(xs, xs));
    __m128 one_minus_ys2 = _mm_sub_ps(one, _mm_mul_ps(ys, ys));
    __m128 zs_add_xi = _mm_add_ps(zs, xi_m128);
    __m128 zs_mul_xi = _mm_mul_ps(zs, xi_m128);
    //    __m128 zs_add_xiSq = _mm_mul_ps(zs_add_xi, zs_add_xi);

    __m128 dx = _mm_mul_ps(_mm_load_ps(buf_warped_dx.data() + i), fxl);
    __m128 dy = _mm_mul_ps(_mm_load_ps(buf_warped_dy.data() + i), fyl);
    __m128 u = _mm_load_ps(buf_warped_u.data() + i);
    __m128 v = _mm_load_ps(buf_warped_v.data() + i);

    acc.UpdateSSEEighted(
        _mm_mul_ps(
            fact_t,
            _mm_sub_ps(
                _mm_mul_ps(_mm_add_ps(_mm_mul_ps(one_minus_xs2, xi_m128), zs),
                           dx),
                _mm_mul_ps(_mm_mul_ps(xsys, xi_m128), dy))),
        _mm_mul_ps(
            fact_t,
            _mm_sub_ps(
                _mm_mul_ps(_mm_add_ps(_mm_mul_ps(one_minus_ys2, xi_m128), zs),
                           dy),
                _mm_mul_ps(_mm_mul_ps(xsys, xi_m128), dx))),
        _mm_mul_ps(
            fact_t,
            _mm_sub_ps(
                _mm_sub_ps(
                    zero,
                    _mm_mul_ps(_mm_mul_ps(xs, _mm_add_ps(one, zs_mul_xi)), dx)),
                _mm_mul_ps(_mm_mul_ps(ys, _mm_add_ps(one, zs_mul_xi)), dy))),
        _mm_mul_ps(
            fact_r,
            _mm_sub_ps(_mm_sub_ps(zero, _mm_mul_ps(xsys, dx)),
                       _mm_mul_ps(_mm_add_ps(one_minus_xs2, zs_mul_xi), dy))),
        _mm_mul_ps(
            fact_r,
            _mm_add_ps(_mm_mul_ps(_mm_add_ps(one_minus_ys2, zs_mul_xi), dx),
                       _mm_mul_ps(xsys, dy))),
        _mm_mul_ps(fact_r,
                   _mm_sub_ps(_mm_mul_ps(_mm_mul_ps(xs, zs_add_xi), dy),
                              _mm_mul_ps(_mm_mul_ps(ys, zs_add_xi), dx))),
        _mm_mul_ps(a, _mm_sub_ps(b0, _mm_load_ps(buf_warped_ref_color.data() +
                                                 i))),  // w.r.t. a_t
        minusOne,                                       // w.r.t. b_t
        _mm_load_ps(buf_warped_residual.data() + i),
        _mm_load_ps(buf_warped_weight.data() + i));
  }

  acc.Finish();
  H_out = acc.H.topLeftCorner<8, 8>().cast<double>() * (1.0f / n);
  b_out = acc.H.topRightCorner<8, 1>().cast<double>() * (1.0f / n);

  H_out.block<8, 3>(0, 0) *= SCALE_XI_TRANS;
  H_out.block<8, 3>(0, 3) *= SCALE_XI_ROT;
  H_out.block<8, 1>(0, 6) *= SCALE_A;
  H_out.block<8, 1>(0, 7) *= SCALE_B;
  H_out.block<3, 8>(0, 0) *= SCALE_XI_TRANS;
  H_out.block<3, 8>(3, 0) *= SCALE_XI_ROT;
  H_out.block<1, 8>(6, 0) *= SCALE_A;
  H_out.block<1, 8>(7, 0) *= SCALE_B;
  b_out.segment<3>(0) *= SCALE_XI_TRANS;
  b_out.segment<3>(3) *= SCALE_XI_ROT;
  b_out.segment<1>(6) *= SCALE_A;
  b_out.segment<1>(7) *= SCALE_B;
}

}  // namespace dsl