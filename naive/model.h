
#pragma once

#include "vec3.h"

struct Triangle {
    Vec3 v0;
    Vec3 v1;
    Vec3 v2;
};

struct Box {
    Vec3 min;
    Vec3 max;
};

struct Model {
    int n_triangles;
    Triangle *triangles;
    Box bounding_box;
};


Model *load_model(const char *filename);
