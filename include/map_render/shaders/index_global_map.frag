/*
 * This file is part of ElasticFusion.
 *
 * Copyright (C) 2015 Imperial College London
 * 
 * The use of the code within this file and all code within files that 
 * make up the software that is ElasticFusion is permitted for 
 * non-commercial purposes only.  The full terms and conditions that 
 * apply to the code within this file are detailed within the LICENSE.txt 
 * file and at <http://www.imperial.ac.uk/dyson-robotics-lab/downloads/elastic-fusion/elastic-fusion-license/> 
 * unless explicitly stated.  By downloading this file you agree to 
 * comply with these terms.
 *
 * If you wish to use any of this code for commercial purposes then 
 * please email researchcontracts.engineering@imperial.ac.uk.
 *
 */

// redundant

#version 330 core

// in vec4 vPosition0;
// in vec4 vColorTime0;
// in vec4 vNormRad0;
// flat in int vertexId;

in vec2 texcoord;
flat in vec4 vPosition1;
flat in vec4 vColorTime1;
flat in vec4 vNormRad1;
flat in int vertexId1;

layout(location = 0) out int FragColor;
layout(location = 1) out vec4 vPosition2;
layout(location = 2) out vec4 vColorTime2;
layout(location = 3) out vec4 vNormRad2;

void main()
{
    if(dot(texcoord, texcoord) > 1.0)
        discard;
    vPosition2 = vPosition1;
    vColorTime2 = vColorTime1;
    vNormRad2 = vNormRad1; 
    FragColor = vertexId1;
}
