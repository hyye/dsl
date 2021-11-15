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

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

in vec4 vPosition0[];
in vec4 vColorTime0[];
in vec4 vNormRad0[];
flat in int vertexId[];

out vec2 texcoord;
flat out vec4 vPosition1;
flat out vec4 vColorTime1;
flat out vec4 vNormRad1;
flat out int vertexId1;

uniform mat4 t_inv;
uniform vec4 cam; //cx, cy, fx, fy
uniform float xi;
uniform float max_half_fov;
uniform float cols;
uniform float rows;
uniform float maxDepth;

vec4 PointToPixel(vec4 vPosHome) {
    float x = 0, y = 0;
    // if(vPosHome.z > maxDepth || vPosHome.z < 0)
    // {
        // x = -10;
        // y = -10;
    // }
    // else
    // {
//        x = ((((cam.z * vPosHome.x) / vPosHome.z) + cam.x) - (cols * 0.5)) / (cols * 0.5);
//        y = ((((cam.w * vPosHome.y) / vPosHome.z) + cam.y) - (rows * 0.5)) / (rows * 0.5);
    vec3 ps = normalize(vPosHome.xyz);
    float deno = ps.z + xi;
    float xs = ps.x;
    float ys = ps.y;
    float xm = xs / deno;
    float ym = ys / deno;
    x = (((cam.z * xm) + cam.x) - (cols * 0.5)) / (cols * 0.5);
    y = (((cam.w * ym) + cam.y) - (rows * 0.5)) / (rows * 0.5);
    // }
    return vec4(x, y, vPosHome.z / maxDepth, 1.0);
}

void main()
{
    vec4 v1, v2, v3, v4;
    vec4 n;

    vec3 x = normalize(vec3((vNormRad0[0].y - vNormRad0[0].z), -vNormRad0[0].x, vNormRad0[0].x)) * vNormRad0[0].w * 1.41421356;
    vec3 y = cross(vNormRad0[0].xyz, x);

    v1 = t_inv * vec4(vPosition0[0].xyz + x, 1.0);
    v2 = t_inv * vec4(vPosition0[0].xyz + y, 1.0);
    v3 = t_inv * vec4(vPosition0[0].xyz - y, 1.0);
    v4 = t_inv * vec4(vPosition0[0].xyz - x, 1.0);

    if(v1.z <= maxDepth && ValidFov(v1.xyz, max_half_fov) &&
       v2.z <= maxDepth && ValidFov(v2.xyz, max_half_fov) &&
       v3.z <= maxDepth && ValidFov(v3.xyz, max_half_fov) &&
       v4.z <= maxDepth && ValidFov(v4.xyz, max_half_fov)) {

        n = vec4(vNormRad0[0].xyz, 1.0);
        
        vPosition1 = vPosition0[0];
        vNormRad1 = vNormRad0[0];
        vColorTime1 = vColorTime0[0];
        vertexId1 = vertexId[0];

        texcoord = vec2(-1.0, -1.0);
        gl_Position = PointToPixel(v1);
        EmitVertex();

        texcoord = vec2(1.0, -1.0);
        gl_Position = PointToPixel(v2);
        EmitVertex();

        texcoord = vec2(-1.0, 1.0);
        gl_Position = PointToPixel(v3);
        EmitVertex();

        texcoord = vec2(1.0, 1.0);
        gl_Position = PointToPixel(v4);
        EmitVertex();
        EndPrimitive();
    }
}
