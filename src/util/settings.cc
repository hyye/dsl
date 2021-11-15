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

#include "util/settings.h"

namespace dsl {

int pyrLevelsUsed = PYR_LEVELS;

float settingKeyframesPerSecond = 0;
// WARNING: change this values to wG, hG related?
float settingMaxShiftWeightT = 0.04f * (640 + 480);
float settingMaxShiftWeightR = 0.0f * (640 + 480);

float settingMaxShiftWeightRT = 0.02f * (640 + 480);
// general weight on threshold, the larger the more KF's are taken (e.g., 2 =
// double the amount of KF's).
float settingKfGlobalWeight = 1;
float settingMaxAffineWeight = 2;

float settingOverallEnergyThWeight = 1;
float settingCoarseCutoffTh = 20;

float settingHuberTh = 9;
bool settingDebugoutRunquiet = true;
int settingAffineOptModeA = 0;
int settingAffineOptModeB = 0;

float settingMinGradHistCut = 0.5;
float settingMinGradHistAdd = 7;
float settingGradDownweightPerLevel = 0.75;
bool settingSelectDirectionDistribution = true;

// immature points per frame
float settingDesiredImmatureDensity = 1500;
// aimed total points in the active window.
float settingDesiredPointDensity = 2000;
// marg a frame if less than X% points remain.
float settingMinPointsRemaining = 0.05;
// marg a frame if factor between intensities to current frame is larger than
// 1/X or X.
float settingMaxLogAffFacInWindow = 0.7;

// Outlier Threshold on photometric energy
// higher -> less strict
float settingOutlierTh = 12 * 12;
// higher -> less strong gradient-based reweighting.
float settingOutlierThSumComponent = 50 * 50;

// (larger = re-track more often)
float settingReTrackThreshold = 1.5;

float settingMaxPixSearch = 0.027;
float settingTraceSlackInterval = 1.5;
float settingTraceStepsize = 1.0;
float settingTraceMinImprovementFactor = 2;
float settingMinTraceTestRadius = 2;
float settingTraceGNThreshold = 0.1;
float settingTraceExtraSlackOnTH = 1.2;
int settingTraceGNIterations = 3;

int settingMinFrames = 5;
int settingMaxFrames = 7;
int settingMinFrameAge = 1;

int settingMaxOptIterations = 6;
int settingMinOptIterations = 1;

int settingGNItsOnPointActivation = 3;

float settingMinIdistHMarg = 50;
float settingMinIdistHAct = 100;
int settingMinGoodActiveResForMarg = 3;
int settingMinGoodResForMarg = 4;

float settingMinTraceQuality = 3;

bool settingMultiThreading = true;

float settingFrameEnergyThConstWeight = 0.5;
float settingFrameEnergyThN = 0.7;
float settingFrameEnergyThFacMedian = 1.5;

float benchmarkInitializerSlackFactor = 1;

bool settingBaseline = false;
bool settigEnableMarginalization = true;
bool settigNoScalingAtOpt = true;

int staticPattern[10][40][2] = {
    {{0, 0},       {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},  // .
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100}},

    {{0, -1},      {-1, 0},      {0, 0},       {1, 0},       {0, 1},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},  // +
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100}},

    {{-1, -1},     {1, 1},       {0, 0},       {-1, 1},      {1, -1},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},  // x
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100}},

    {{-1, -1},     {-1, 0},      {-1, 1},      {-1, 0},
     {0, 0},       {0, 1},       {1, -1},      {1, 0},
     {1, 1},       {-100, -100},  // full-tight
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}},

    {{0, -2},      {-1, -1},     {1, -1},      {-2, 0},
     {0, 0},       {2, 0},       {-1, 1},      {1, 1},
     {0, 2},       {-100, -100},  // full-spread-9
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}},

    {{0, -2},      {-1, -1},     {1, -1},      {-2, 0},
     {0, 0},       {2, 0},       {-1, 1},      {1, 1},
     {0, 2},       {-2, -2},  // full-spread-13
     {-2, 2},      {2, -2},      {2, 2},       {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}},

    {{-2, -2},     {-2, -1},     {-2, -0},     {-2, 1},
     {-2, 2},      {-1, -2},     {-1, -1},     {-1, -0},
     {-1, 1},      {-1, 2},  // full-25
     {-0, -2},     {-0, -1},     {-0, -0},     {-0, 1},
     {-0, 2},      {+1, -2},     {+1, -1},     {+1, -0},
     {+1, 1},      {+1, 2},      {+2, -2},     {+2, -1},
     {+2, -0},     {+2, 1},      {+2, 2},      {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}},

    {{0, -2},      {-1, -1},     {1, -1},      {-2, 0},
     {0, 0},       {2, 0},       {-1, 1},      {1, 1},
     {0, 2},       {-2, -2},  // full-spread-21
     {-2, 2},      {2, -2},      {2, 2},       {-3, -1},
     {-3, 1},      {3, -1},      {3, 1},       {1, -3},
     {-1, -3},     {1, 3},       {-1, 3},      {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}},

    {{0, -2},      {-1, -1},     {1, -1},      {-2, 0},
     {0, 0},       {2, 0},       {-1, 1},      {0, 2},
     {-100, -100}, {-100, -100},  // 8 for SSE efficiency
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}},

    {{-4, -4},     {-4, -2},     {-4, -0},     {-4, 2},
     {-4, 4},      {-2, -4},     {-2, -2},     {-2, -0},
     {-2, 2},      {-2, 4},  // full-45-SPREAD
     {-0, -4},     {-0, -2},     {-0, -0},     {-0, 2},
     {-0, 4},      {+2, -4},     {+2, -2},     {+2, -0},
     {+2, 2},      {+2, 4},      {+4, -4},     {+4, -2},
     {+4, -0},     {+4, 2},      {+4, 4},      {-200, -200},
     {-200, -200}, {-200, -200}, {-200, -200}, {-200, -200},
     {-200, -200}, {-200, -200}, {-200, -200}, {-200, -200},
     {-200, -200}, {-200, -200}, {-200, -200}, {-200, -200},
     {-200, -200}, {-200, -200}},
};

int staticPatternNum[10] = {1, 5, 5, 9, 9, 13, 25, 21, 8, 25};

int staticPatternPadding[10] = {1, 1, 1, 1, 2, 2, 2, 3, 2, 4};

}