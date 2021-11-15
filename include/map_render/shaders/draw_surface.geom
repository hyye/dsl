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

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

uniform float threshold;
uniform float signMult;

in vec4 vColor[];
in vec4 vPosition[];
in vec4 vNormRad[];
in mat4 vMVP[];
in int colorType0[];

out vec3 vColor0;
out vec2 texcoord;
out float radius;
flat out int unstablePoint;

#include "color.glsl"

void main()
{
    vec3 v;
    vec3 n;

    vec4 localPosition = vMVP[0] * vec4(vPosition[0].xyz, 1.0);
    if (localPosition.z >= 0.5) {
        if(colorType0[0] == 1) {
            vColor0 = vNormRad[0].xyz;
            if (vColor0.z < 0) {
                vColor0 = -vColor0;
            }
        } else if(colorType0[0] == 2) {
            if (vColor[0].y == 1) {
                  vColor0 = decodeColor(vColor[0].x);
                  // vColor0 = vec3(1.0f, 0, 0);
            } else {
                vColor0 = vec3(.0f, .0f, .0f);
            }
        } else {
            // WARNING: temporal use for intensity // * 2 for visualization only
            vColor0 = vec3(vColor[0].w) / 255.0;
//            vColor0 = (vec3(.5f, .5f, .5f) * abs(dot(vNormRad[0].xyz, vec3(1.0, 1.0, 1.0)))) + vec3(0.1f, 0.1f, 0.1f);
        }
    //    vec4 vNormRad[1];
    //    vNormRad[0].xyz = normalize(vPosition[0].xyz);
        radius = vNormRad[0].w;

        vec3 x = normalize(vec3((vNormRad[0].y - vNormRad[0].z), -vNormRad[0].x, vNormRad[0].x)) * vNormRad[0].w * 1.41421356;

        vec3 y = cross(vNormRad[0].xyz, x);

        unstablePoint = 1;
        n = vNormRad[0].xyz;

        texcoord = vec2(-1.0, -1.0);
        gl_Position = vMVP[0] * vec4(vPosition[0].xyz + x, 1.0);
        v = vPosition[0].xyz + x;
        EmitVertex();

        texcoord = vec2(1.0, -1.0);
        gl_Position = vMVP[0] * vec4(vPosition[0].xyz + y, 1.0);
        v = vPosition[0].xyz + y;
        EmitVertex();

        texcoord = vec2(-1.0, 1.0);
        gl_Position = vMVP[0] * vec4(vPosition[0].xyz - y, 1.0);
        v = vPosition[0].xyz - y;
        EmitVertex();

        texcoord = vec2(1.0, 1.0);
        gl_Position = vMVP[0] * vec4(vPosition[0].xyz - x, 1.0);
        v = vPosition[0].xyz - x;
        EmitVertex();
        EndPrimitive();
    }

}
