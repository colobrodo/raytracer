
// to use default implementation of tiny obj loader
#define TINYOBJLOADER_IMPLEMENTATION
#include "../include/tiny_obj_loader.h"

#include "./model.h"


Model *load_model(const char *filename) {
    tinyobj::attrib_t attributes;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn;
    std::string err;
    bool success = tinyobj::LoadObj(&attributes, &shapes, &materials, &warn, &err, filename);

    printf("Loading model %s\n", filename);

    if (!warn.empty()) {
        printf("warning loading the obj file: %s\n", warn.c_str());
    }
    if (!err.empty()) {
        printf("Error trying to load obj file: %s\n", err.c_str());
        return nullptr;
    }

    if(!success) {
        printf("Error trying to load obj file: %s\n", filename);
        return nullptr;
    }

    // printf("# of vertices  = %d\n", (int)(attributes.vertices.size()) / 3);
    // printf("# of normals   = %d\n", (int)(attributes.normals.size()) / 3);

    // allocate triangles
    auto total_triangles = 0;
    for(auto &shape: shapes) {
        total_triangles += shape.mesh.indices.size();
    }
    Triangle *triangles = (Triangle*) malloc(total_triangles * sizeof(Triangle));

    Box box {{INFINITY, INFINITY, INFINITY}, {-INFINITY, -INFINITY, -INFINITY}};

    auto update_bounding_box = [&](const Vec3 &point){
        if(box.min.x > point.x) {
            box.min.x = point.x;
        }
        if(box.min.y > point.y) {
            box.min.y = point.y;
        }
        if(box.min.z > point.z) {
            box.min.z = point.z;
        }

        if(box.max.x < point.x) {
            box.max.x = point.x;
        }
        if(box.max.y < point.y) {
            box.max.y = point.y;
        }
        if(box.max.z < point.z) {
            box.max.z = point.z;
        }
    };

    for(auto &shape: shapes) {
        for (size_t f = 0; f < shape.mesh.indices.size(); f += 3) {
            // Get the three vertex indexes and coordinates
            Triangle triangle;

            // Get the three indexes of the face (all faces are triangular)
            for (int k = 0; k < 3; k++) {
                tinyobj::index_t index = shape.mesh.indices[f + k];
                // assert(index.vertex_index >= 0);
                const float scale = .3f;
                float x = attributes.vertices[3 * index.vertex_index + 0] * scale,
                      y = attributes.vertices[3 * index.vertex_index + 1] * scale,
                      z = attributes.vertices[3 * index.vertex_index + 2] * scale - 3;
                if(k == 0) {
                  triangle.v0.x = x;
                  triangle.v0.y = y;
                  triangle.v0.z = z;
                } else if(k == 1) {
                  triangle.v1.x = x;
                  triangle.v1.y = y;
                  triangle.v1.z = z;
                } else /*k == 2*/ {
                  triangle.v2.x = x;
                  triangle.v2.y = y;
                  triangle.v2.z = z;
                }

            }
            int i = f / 3;
            triangles[i].v0 = triangle.v0;
            triangles[i].v1 = triangle.v1;
            triangles[i].v2 = triangle.v2;
            update_bounding_box(triangles[i].v0);
            update_bounding_box(triangles[i].v1);
            update_bounding_box(triangles[i].v2);
        }
    }

    // DEBUG:
    /*
    printf("Bounding box:\n");
    printf("\t min: %f %f %f\n", box.min.x, box.min.y, box.min.z);
    printf("\t max: %f %f %f\n", box.max.x, box.max.y, box.max.z);
    */

    Model *model = new Model{total_triangles, triangles, box};
    return model;
}
