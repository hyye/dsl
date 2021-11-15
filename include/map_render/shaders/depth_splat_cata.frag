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

#version 330 core
#include "fov.glsl"

uniform vec4 cam; //cx, cy, fx, fy
uniform float xi;
uniform float max_half_fov;
uniform float maxDepth;

in vec4 position;
in vec4 normRad;

out float FragColor;

void main()
{
//    vec3 l = normalize(vec3((vec2(gl_FragCoord) - cam.xy) / cam.zw, 1.0f));
    vec3 mu = vec3((vec2(gl_FragCoord) - cam.xy) / cam.zw, 1.0f);
    float d2 = mu.x * mu.x + mu.y * mu.y;
    float factor = (xi + sqrt(1 + (1 - xi * xi) * d2)) / (d2 + 1);
    vec3 l = normalize(vec3(vec2(mu.xy) * factor, factor - xi));

    vec3 corrected_pos = (dot(position.xyz, normRad.xyz) / dot(l, normRad.xyz)) * l;

    //check if the intersection is inside the surfel
    float sqrRad = pow(normRad.w, 2);
    vec3 diff = corrected_pos - position.xyz;

    if(dot(diff, diff) > sqrRad || !ValidFov(corrected_pos, max_half_fov))
    {
        discard;
    }

//    vec3 corrected_pos = position.xyz;

//    FragColor = corrected_pos.z;
    FragColor = length(corrected_pos.xyz);

    gl_FragDepth = (corrected_pos.z / (2 * maxDepth)) + 0.5f;
}
