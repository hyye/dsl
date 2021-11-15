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

#include "full_system/ef_struct.h"
#include "full_system/hessian_blocks.h"

namespace dsl {

void EfResidual::FixLinearizationF(EnergyFunction *ef) {
  // FIXME: FixLinearizationF, marginalized residual calucated here!!!
  is_linearized = true;
}

void EfResidual::TakeDataF() {
  // FIXME: TakeDataF, update jacobian and swap with pfr->J!!!
}

EnergyFunction::EnergyFunction() {
  num_frames = num_residuals = num_points = 0;
}

void EnergyFunction::InsertFrame(FrameHessian *fh, CalibHessian &HCalib) {
  std::unique_ptr<EfFrame> eff = std::make_unique<EfFrame>(fh);
  fh->ef_frame = eff.get();
  ef_frames.emplace_back(std::move(eff));
  ++num_frames;
}

void EnergyFunction::InsertPoint(std::unique_ptr<PointHessian> &ph) {
  std::unique_ptr<EfPoint> efp =
      std::make_unique<EfPoint>(&(*ph), ph->host->ef_frame);
  efp->idx_in_points = ph->host->ef_frame->ef_points.size();
  ph->ef_point = efp.get();
  ph->host->ef_frame->ef_points.emplace_back(std::move(efp));
  ph->host->point_hessians.emplace_back(std::move(ph));
  ++num_points;
  ef_indices_valid = false;
}

void EnergyFunction::InsertResidual(PointFrameResidual *r) {
  std::unique_ptr<EfResidual> efr = std::make_unique<EfResidual>(
      r, r->point->ef_point, r->host->ef_frame, r->target->ef_frame);
  efr->idx_in_all = r->point->ef_point->ef_residuals.size();
  r->ef_residual = efr.get();
  r->point->ef_point->ef_residuals.emplace_back(std::move(efr));

  // TODO: connectivity_map

  ++num_residuals;
}

void EnergyFunction::MakeIdx() {
  for (unsigned long idx = 0; idx < ef_frames.size(); ++idx) {
    ef_frames[idx]->idx = idx;
  }

  all_points.clear();

  for (std::unique_ptr<EfFrame> &f : ef_frames) {
    for (std::unique_ptr<EfPoint> &p : f->ef_points) {
      all_points.push_back(p.get());
      for (std::unique_ptr<EfResidual> &r : p->ef_residuals) {
        // NOTE: update residual's frame id in new frames
        r->host_idx = r->host->idx;
        r->target_idx = r->target->idx;
      }
    }
  }

  ef_indices_valid = true;
}

void EnergyFunction::DropPointsF() {
  for (std::unique_ptr<EfFrame> &f : ef_frames) {
    for (int i = 0; i < (int)f->ef_points.size(); ++i) {
      std::unique_ptr<EfPoint> &p = f->ef_points[i];
      if (p->state_flag == EfPointStatus::DROP) {
        RemovePoint(p.get());
        --i;
      }
    }
  }

  ef_indices_valid = false;
  MakeIdx();
}

void EnergyFunction::RemovePoint(EfPoint *efp) {
  for (int i = 0; i < efp->ef_residuals.size();) {
    // FIXME FIXME:HUGE ERROR
    std::unique_ptr<EfResidual> &r = efp->ef_residuals[0];
    DropResidual(r.get());
  }

  EfFrame *h = efp->host;
  int buf_idx = efp->idx_in_points;
  std::swap(h->ef_points[buf_idx], h->ef_points.back());
  h->ef_points[buf_idx]->idx_in_points = buf_idx;
  efp->point->ef_point = nullptr;

  ef_indices_valid = false;

  h->ef_points.pop_back();
  --num_points;
}

void EnergyFunction::DropResidual(EfResidual *efr) {
  EfPoint *p = efr->point;
  assert(efr == p->ef_residuals[efr->idx_in_all].get());

  int buf_idx = efr->idx_in_all;
  swap(p->ef_residuals[buf_idx], p->ef_residuals.back());
  p->ef_residuals[buf_idx]->idx_in_all = buf_idx;

  if (efr->IsActive())
    efr->host->frame->shell->statistics_good_res++;
  else
    efr->host->frame->shell->statistics_outlier_res++;

  // TODO: connectivity_map

  efr->pfr->ef_residual = nullptr;

  p->ef_residuals.pop_back();  // delete
  --num_residuals;
}

void EnergyFunction::MarginalizeFrame(EfFrame *eff) {
  // FIXME: Matrix shift etc...
  DeleteOutOrder<EfFrame>(ef_frames, eff);
  MakeIdx();
  --num_frames;
}

void EnergyFunction::MarginalizePointsF() {
  // FIXME: Matrix shift etc...
  int num_marg = 0;
  for (std::unique_ptr<EfFrame> &f : ef_frames) {
    for (int i = 0; i < (int)f->ef_points.size(); i++) {
      EfPoint *p = f->ef_points[i].get();
      if (p->state_flag == EfPointStatus::MARGINALIZE) {
        RemovePoint(p);
        --i;
        ++num_marg;
      }
    }
  }
  LOG(INFO) << "REMOVE num_marg: " << num_marg;
}

void EnergyFunction::SetAdjointsF(CalibHessian &HCalib) {
  ad_host.resize(num_frames * num_frames);
  ad_target.resize(num_frames * num_frames);

  for (int h = 0; h < num_frames; h++)
    for (int t = 0; t < num_frames; t++) {
      FrameHessian *host = ef_frames[h]->frame;
      FrameHessian *target = ef_frames[t]->frame;

      SE3 host_to_target =
          target->GetWorldToCamEvalPT() * host->GetWorldToCamEvalPT().inverse();

      Mat88 AH = Mat88::Identity();
      Mat88 AT = Mat88::Identity();

      // NOTE: sth like the jacobian, the transpose is to make it suitable for
      // x^T * J^T = (J * x)^T
      AH.topLeftCorner<6, 6>() = -host_to_target.Adj().transpose();
      AT.topLeftCorner<6, 6>() = Mat66::Identity();

      Vec2f affLL = AffLight::FromToVecExposure(
                        host->exposure, target->exposure,
                        host->GetAffLightZero(), target->GetAffLightZero())
                        .cast<float>();
      // NOTE: like jacobian of a,b in (11), i--host, j--target, but with exp
      // and multiplication; the minus differs
      /// Here the adjoint matrix is opposite, corresponding to the opposite aff
      /// jacobian
      AT(6, 6) = -affLL[0];  // aj
      AH(6, 6) = affLL[0];   // ai
      AT(7, 7) = -1;         // bj
      AH(7, 7) = affLL[0];   // bi

      AH.block<3, 8>(0, 0) *= SCALE_XI_TRANS;
      AH.block<3, 8>(3, 0) *= SCALE_XI_ROT;
      AH.block<1, 8>(6, 0) *= SCALE_A;
      AH.block<1, 8>(7, 0) *= SCALE_B;
      AT.block<3, 8>(0, 0) *= SCALE_XI_TRANS;
      AT.block<3, 8>(3, 0) *= SCALE_XI_ROT;
      AT.block<1, 8>(6, 0) *= SCALE_A;
      AT.block<1, 8>(7, 0) *= SCALE_B;

      ad_host[h + t * num_frames] = AH;
      ad_target[h + t * num_frames] = AT;
    }

  ad_hostf.resize(num_frames * num_frames);
  ad_targetf.resize(num_frames * num_frames);

  for (int h = 0; h < num_frames; h++) {
    for (int t = 0; t < num_frames; t++) {
      ad_hostf[h + t * num_frames] = ad_host[h + t * num_frames].cast<float>();
      ad_targetf[h + t * num_frames] =
          ad_target[h + t * num_frames].cast<float>();
    }
  }

  ef_adjoints_valid = true;
}

}