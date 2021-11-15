bool ValidFov(vec3 p, float max_half_fov) {
    float s = sqrt(p.x * p.x + p.y * p.y);
    float z = p.z;
    return abs(atan(s, z)) < max_half_fov;
}