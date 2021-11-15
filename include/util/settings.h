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

#ifndef DSL_SETTINGS_H_
#define DSL_SETTINGS_H_

#include <cerrno>
namespace dsl {

#define PYR_LEVELS 6
extern int pyrLevelsUsed;

extern float settingKeyframesPerSecond;
extern float settingMaxShiftWeightT;
extern float settingMaxShiftWeightR;
extern float settingMaxShiftWeightRT;
extern float settingKfGlobalWeight;
extern float settingMaxAffineWeight;

extern float settingOverallEnergyThWeight;
extern float settingCoarseCutoffTh;

extern float settingHuberTh;
extern bool settingDebugoutRunquiet;
extern int settingAffineOptModeA;
extern int settingAffineOptModeB;

extern float settingMinGradHistCut;
extern float settingMinGradHistAdd;
extern float settingGradDownweightPerLevel;
extern bool settingSelectDirectionDistribution;

extern float settingDesiredImmatureDensity;
extern float settingDesiredPointDensity;
extern float settingMinPointsRemaining;
extern float settingMaxLogAffFacInWindow;

extern float settingOutlierTh;
extern float settingOutlierThSumComponent;

extern float settingReTrackThreshold;

extern float settingMaxPixSearch;
extern float settingTraceSlackInterval;
extern float settingTraceStepsize;
extern float settingTraceMinImprovementFactor;
extern float settingMinTraceTestRadius;
extern float settingTraceGNThreshold;
extern float settingTraceExtraSlackOnTH;
extern int settingTraceGNIterations;

extern int settingMinFrames;
extern int settingMaxFrames;
extern int settingMinFrameAge;

// max GN iterations
extern int settingMaxOptIterations;
extern int settingMinOptIterations;

extern int settingGNItsOnPointActivation;

extern float settingMinIdistHMarg;
extern float settingMinIdistHAct;
extern int settingMinGoodActiveResForMarg;
extern int settingMinGoodResForMarg;

extern float settingMinTraceQuality;

extern bool settingMultiThreading;

extern float benchmarkInitializerSlackFactor;

extern float settingFrameEnergyThConstWeight;
extern float settingFrameEnergyThN;
extern float settingFrameEnergyThFacMedian;

extern bool settingBaseline;
extern bool settigEnableMarginalization;
extern bool settigNoScalingAtOpt;

extern int staticPattern[10][40][2];
extern int staticPatternNum[10];
extern int staticPatternPadding[10];

#define patternNum 8
#define patternP staticPattern[8]
#define patternPadding 2

}  // namespace dsl

#endif  // DSL_SETTINGS_H_
