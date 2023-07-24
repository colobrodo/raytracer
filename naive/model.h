#pragma once


#include "vec3.h"

using Triangle = Vec3[3]; 

struct Box
{
    Vec3 min;
    Vec3 max;
};

struct Model {
    int n_triangles;
    Triangle *triangles;
    Box bounding_box;
};


Model *load_model(const char *filename);
