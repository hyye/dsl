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
// Created by hyye on 11/11/19.
//

#include "full_system/pixel_selector.h"

namespace dsl {

int CalcHistQuantil(std::vector<int>& hist, float below) {
  int th = hist[0] * below + 0.5f;
  // WARNING: 90?
  for (int i = 0; i < 90; i++) {
    th -= hist[i + 1];
    if (th < 0) return i;
  }
  return 90;
}

PixelSelector::PixelSelector(int w, int h) {
  random_pattern = std::vector<unsigned char>(w * h);
  std::srand(3141592);  // want to be deterministic, follows DSO
  for (int i = 0; i < w * h; i++) random_pattern[i] = rand() & 0xFF;

  current_potential = 3;

  grad_hist = std::vector<int>(100 * (1 + w / 32) * (1 + h / 32));
  ths = std::vector<float>(100 * (1 + w / 32) * (1 + h / 32));
  ths_smoothed = std::vector<float>(100 * (1 + w / 32) * (1 + h / 32));

  grad_hist_frame = 0;
}

void PixelSelector::MakeHists(const FrameHessian& fh) {
  grad_hist_frame = &fh;
  const float* mapmax0 = fh.abs_sq_grad[0].data();

  // NOTE: why not in the initialization?
  int w = wG[0];
  int h = hG[0];

  int w32 = w / 32;
  int h32 = h / 32;
  ths_step = w32;

  for (int y = 0; y < h32; y++) {
    for (int x = 0; x < w32; x++) {
      const float* map0 = mapmax0 + 32 * x + 32 * y * w;
      std::vector<int>& hist0 = grad_hist;  // + 50*(x+y*w32);

      std::fill(hist0.begin(), hist0.end(), 0);
      // memset(hist0.data(), 0, sizeof(int) * 50);

      for (int j = 0; j < 32; j++)
        for (int i = 0; i < 32; i++) {
          int it = i + 32 * x;
          int jt = j + 32 * y;
          if (it > w - 2 || jt > h - 2 || it < 1 || jt < 1) continue;
          int g = sqrtf(map0[i + j * w]);
          if (g > 48) g = 48;
          hist0[g + 1]++;
          hist0[0]++;
        }

      // NOTE: threshold for theshold of 32x32 blocks -- median +
      // setting_minGradHistAdd
      ths[x + y * w32] =
          CalcHistQuantil(hist0, settingMinGradHistCut) + settingMinGradHistAdd;
    }
  }

  for (int y = 0; y < h32; y++) {
    for (int x = 0; x < w32; x++) {
      float sum = 0, num = 0;
      if (x > 0) {
        if (y > 0) {
          num++;
          sum += ths[x - 1 + (y - 1) * w32];
        }
        if (y < h32 - 1) {
          num++;
          sum += ths[x - 1 + (y + 1) * w32];
        }
        num++;
        sum += ths[x - 1 + (y)*w32];
      }

      if (x < w32 - 1) {
        if (y > 0) {
          num++;
          sum += ths[x + 1 + (y - 1) * w32];
        }
        if (y < h32 - 1) {
          num++;
          sum += ths[x + 1 + (y + 1) * w32];
        }
        num++;
        sum += ths[x + 1 + (y)*w32];
      }

      if (y > 0) {
        num++;
        sum += ths[x + (y - 1) * w32];
      }
      if (y < h32 - 1) {
        num++;
        sum += ths[x + (y + 1) * w32];
      }
      num++;
      sum += ths[x + y * w32];

      // NOTE: squared smoothed by 3x3 averaging
      ths_smoothed[x + y * w32] = (sum / num) * (sum / num);
    }
  }
}

int PixelSelector::MakeMaps(const FrameHessian& fh, std::vector<float>& map_out,
                            float density, int recursions_left,
                            float th_factor) {
  float num_have = 0;
  float num_want = density;
  float quotient;
  int ideal_potential = current_potential;

  if (&fh != grad_hist_frame) {
    MakeHists(fh);
  }

  // NOTE: map_out is the PixelSelectorStatus, select in three levels as n[0],
  // n[1], n[2], (n2,n3,n4) select!
  Eigen::Vector3i n = Select(fh, map_out, current_potential, th_factor);
  num_have = n[0] + n[1] + n[2];
  quotient = num_want / num_have;

  // by default we want to over-sample by 40% just to be sure.
  float K = num_have * (current_potential + 1) * (current_potential + 1);
  ideal_potential = sqrtf(K / num_want) - 1;  // round down.
  if (ideal_potential < 1) ideal_potential = 1;

  if (recursions_left > 0 && quotient > 1.25 && current_potential > 1) {
    // re-sample to get more points!
    // potential needs to be smaller
    if (ideal_potential >= current_potential) {
      ideal_potential = current_potential - 1;
    }
    current_potential = ideal_potential;
    return MakeMaps(fh, map_out, density, recursions_left - 1, th_factor);
  } else if (recursions_left > 0 && quotient < 0.25) {
    // re-sample to get less points!
    if (ideal_potential <= current_potential) {
      ideal_potential = current_potential + 1;
    }
    current_potential = ideal_potential;
    return MakeMaps(fh, map_out, density, recursions_left - 1, th_factor);
  }

  int num_have_sub = num_have;
  if (quotient < 0.95) {
    int wh = wG[0] * hG[0];
    int rn = 0;
    unsigned char char_th = 255 * quotient;
    for (int i = 0; i < wh; i++) {
      if (map_out[i] != 0) {
        if (random_pattern[rn] > char_th) {
          map_out[i] = 0;
          num_have_sub--;
        }
        rn++;
      }
    }
  }
  current_potential = ideal_potential;
  return num_have_sub;
}

Eigen::Vector3i PixelSelector::Select(const FrameHessian& fh,
                                      std::vector<float>& map_out, int pot,
                                      float th_factor) {
  const Eigen::Vector3f* const map0 = fh.dI;

  const std::vector<float>& mapmax0 = fh.abs_sq_grad[0];
  const std::vector<float>& mapmax1 = fh.abs_sq_grad[1];
  const std::vector<float>& mapmax2 = fh.abs_sq_grad[2];

  int w = wG[0];
  int w1 = wG[1];
  int w2 = wG[2];
  int h = hG[0];

  const Vec2f directions[16] = {
      Vec2f(0, 1.0000),      Vec2f(0.3827, 0.9239),  Vec2f(0.1951, 0.9808),
      Vec2f(0.9239, 0.3827), Vec2f(0.7071, 0.7071),  Vec2f(0.3827, -0.9239),
      Vec2f(0.8315, 0.5556), Vec2f(0.8315, -0.5556), Vec2f(0.5556, -0.8315),
      Vec2f(0.9808, 0.1951), Vec2f(0.9239, -0.3827), Vec2f(0.7071, -0.7071),
      Vec2f(0.5556, 0.8315), Vec2f(0.9808, -0.1951), Vec2f(1.0000, 0.0000),
      Vec2f(0.1951, -0.9808)};

  std::fill(map_out.begin(), map_out.end(), 0);

  float dw1 = settingGradDownweightPerLevel;
  float dw2 = dw1 * dw1;

  int n3 = 0, n2 = 0, n4 = 0;
  for (int y4 = 0; y4 < h; y4 += (4 * pot))
    for (int x4 = 0; x4 < w; x4 += (4 * pot)) {
      int my3 = std::min((4 * pot), h - y4);
      int mx3 = std::min((4 * pot), w - x4);
      int best_idx4 = -1;
      float best_val4 = 0;
      Vec2f dir4 = directions[random_pattern[n2] & 0xF];
      for (int y3 = 0; y3 < my3; y3 += (2 * pot))
        for (int x3 = 0; x3 < mx3; x3 += (2 * pot)) {
          int x34 = x3 + x4;
          int y34 = y3 + y4;
          int my2 = std::min((2 * pot), h - y34);
          int mx2 = std::min((2 * pot), w - x34);
          int best_idx3 = -1;
          float best_val3 = 0;
          Vec2f dir3 = directions[random_pattern[n2] & 0xF];
          for (int y2 = 0; y2 < my2; y2 += pot)
            for (int x2 = 0; x2 < mx2; x2 += pot) {
              int x234 = x2 + x34;
              int y234 = y2 + y34;
              int my1 = std::min(pot, h - y234);
              int mx1 = std::min(pot, w - x234);
              int best_idx2 = -1;
              float best_val2 = 0;
              Vec2f dir2 = directions[random_pattern[n2] & 0xF];
              for (int y1 = 0; y1 < my1; y1 += 1)
                for (int x1 = 0; x1 < mx1; x1 += 1) {
                  assert(x1 + x234 < w);
                  assert(y1 + y234 < h);
                  int idx = x1 + x234 + w * (y1 + y234);
                  int xf = x1 + x234;
                  int yf = y1 + y234;

                  if (xf < 4 || xf >= w - 5 || yf < 4 || yf > h - 4) continue;

                  // NOTE: pixel threhold, which is from the block histogram
                  // calculation
                  float pixel_th0 =
                      ths_smoothed[(xf >> 5) + (yf >> 5) * ths_step];
                  float pixel_th1 = pixel_th0 * dw1;
                  float pixel_th2 = pixel_th1 * dw2;

                  // NOTE: ag0 - intensity
                  float ag0 = mapmax0[idx];
                  if (ag0 > pixel_th0 * th_factor) {
                    // NOTE: ag0d - gradient
                    Vec2f ag0d = map0[idx].tail<2>();
                    float dir_norm = fabsf((float)(ag0d.dot(dir2)));
                    if (!settingSelectDirectionDistribution) dir_norm = ag0;

                    if (dir_norm > best_val2) {
                      best_val2 = dir_norm;
                      best_idx2 = idx;
                      best_idx3 = -2;
                      best_idx4 = -2;
                    }
                  }
                  if (best_idx3 == -2) continue;

                  // NOTE: if not find best_val2

                  float ag1 = mapmax1[(int)(xf * 0.5f + 0.25f) +
                                      (int)(yf * 0.5f + 0.25f) * w1];
                  if (ag1 > pixel_th1 * th_factor) {
                    Vec2f ag0d = map0[idx].tail<2>();
                    float dir_norm = fabsf((float)(ag0d.dot(dir3)));
                    if (!settingSelectDirectionDistribution) dir_norm = ag1;

                    if (dir_norm > best_val3) {
                      best_val3 = dir_norm;
                      best_idx3 = idx;
                      best_idx4 = -2;
                    }
                  }
                  if (best_idx4 == -2) continue;

                  float ag2 = mapmax2[(int)(xf * 0.25f + 0.125) +
                                      (int)(yf * 0.25f + 0.125) * w2];
                  if (ag2 > pixel_th2 * th_factor) {
                    Vec2f ag0d = map0[idx].tail<2>();
                    float dir_norm = fabsf((float)(ag0d.dot(dir4)));
                    if (!settingSelectDirectionDistribution) dir_norm = ag2;

                    if (dir_norm > best_val4) {
                      best_val4 = dir_norm;
                      best_idx4 = idx;
                    }
                  }
                }

              if (best_idx2 > 0) {
                map_out[best_idx2] = 1;
                best_val3 = 1e10;
                n2++;
              }
            }

          if (best_idx3 > 0) {
            map_out[best_idx3] = 2;
            best_val4 = 1e10;
            n3++;
          }
        }

      if (best_idx4 > 0) {
        map_out[best_idx4] = 4;
        n4++;
      }
    }

  return Eigen::Vector3i(n2, n3, n4);
}

}