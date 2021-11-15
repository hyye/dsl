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

in vec3 vColor0;
in vec2 texcoord;
in float radius;
flat in int unstablePoint;

out vec4 FragColor;

void main()
{
    if(dot(texcoord, texcoord) > 1.0)
        discard;

    if (vColor0.x == .0f && vColor0.y == 0 && vColor0.z == 0) {
        FragColor = vec4(.0f, .0f, .0f, .0f);
    } else {
        FragColor = vec4(vColor0, 0.8f);
    }
    // FragColor = vec4(vColor0, 0.8f);

   	gl_FragDepth = gl_FragCoord.z;
}
