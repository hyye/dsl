/*
 * This file is part of li_cam.
 *
 * Copyright (C) 2019 HKUST
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
in vec4 colTime;

layout(location = 0) out vec4 intensity;

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


    intensity = vec4(colTime.w / 255.0);
    intensity.w = 1.0;

    gl_FragDepth = (corrected_pos.z / (2 * maxDepth)) + 0.5f;
}
