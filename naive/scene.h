#pragma once

#include <cmath>
#include <vector>

#include "./vec3.h"
#include "./model.h"

#define EPSILON (0.e-3f)

struct Ray {
    Vec3 origin;
    Vec3 direction;

    Vec3 at(float t) const {
        return origin + direction * t;
    }

};

enum MaterialType {
    METAL,
    PLASTIC,
};

struct Material {
    Vec3 pigment;
    MaterialType type = PLASTIC;
};

struct Sphere {
    Vec3 center;
    float radius;
};

struct Plane {
    Vec3 center;
    Vec3 normal;
};

enum SolidType {
    SPHERE,
    PLANE,
    MODEL,
};

struct Solid {
    SolidType type;
    union {
        Sphere sphere;
        Plane plane;
        Model model;
    };
    Material material;

    float hit(const Ray &ray, Vec3 &normal) const {
        auto box_collision = [] (const Box &box, const Ray &ray) {
            Vec3 dirfrac;
            dirfrac.x = 1.0f / ray.direction.x;
            dirfrac.y = 1.0f / ray.direction.y;
            dirfrac.z = 1.0f / ray.direction.z;
            // lb is the corner of AABB with minimal coordinates - left bottom, rt is maximal corner
            // r.org is origin of ray
            float t1 = (box.min.x - ray.origin.x)*dirfrac.x;
            float t2 = (box.max.x - ray.origin.x)*dirfrac.x;
            float t3 = (box.min.y - ray.origin.y)*dirfrac.y;
            float t4 = (box.max.y - ray.origin.y)*dirfrac.y;
            float t5 = (box.min.z - ray.origin.z)*dirfrac.z;
            float t6 = (box.max.z - ray.origin.z)*dirfrac.z;

            float tmin = __max(__max(__min(t1, t2), __min(t3, t4)), __min(t5, t6));
            float tmax = __min(__min(__max(t1, t2), __max(t3, t4)), __max(t5, t6));

            // if tmax < 0, ray (line) is intersecting AABB, but the whole AABB is behind us
            if (tmax < 0) {
                return false;
            }

            // if tmin > tmax, ray doesn't intersect AABB
            if (tmin > tmax) {
                return false;
            }

            return true;

        };

        switch (type) {
            case SPHERE: {
                auto oc = ray.origin - sphere.center;
                auto a = ray.direction.dot(ray.direction);
                auto b = 2.0 * oc.dot(ray.direction);
                auto c = oc.dot(oc) - sphere.radius * sphere.radius;
                auto discriminant = b * b - 4 * a * c;

                if(discriminant < 0) {
                    return -1;
                } 
                auto t = (-b - sqrt(discriminant) ) / (2.f * a);
                normal = ray.at(t) - sphere.center.normalize();
                return t;
            }
            case PLANE: {
                auto dv = plane.normal.dot(ray.direction);
                if(abs(dv) < EPSILON) {
                    return -1;
                }
                auto d2 = (plane.center - ray.origin).dot(plane.normal);
                auto t = d2 / dv;
                if (t < EPSILON) {
                    return -1;
                }
                normal = plane.normal;
                return t;
            }
            case MODEL: {
                auto closest_t = -1.f;
                if(!box_collision(model.bounding_box, ray)) {
                    return closest_t;
                }
                for(int i = 0; i < model.n_triangles; i++) {
                    auto triangle = model.triangles[i];
                    Vec3 v0 = triangle.v0,
                        v1 = triangle.v1,
                        v2 = triangle.v2;
                    Vec3 v0v1 = v1 - v0;
                    Vec3 v0v2 = v2 - v0;
                    Vec3 pvec = ray.direction.cross(v0v2);
                    float det = v0v1.dot(pvec);
                    // ray and triangle are parallel if det is close to 0
                    if (fabs(det) < EPSILON) continue;
                    float invDet = 1 / det;

                    Vec3 tvec = ray.direction - v0;
                    auto u = tvec.dot(pvec) * invDet;
                    if (u < 0 || u > 1) continue;

                    Vec3 qvec = tvec.cross(v0v1);
                    auto v = ray.direction.dot(qvec) * invDet;
                    if (v < 0 || u + v > 1) continue;
                    
                    auto t = v0v2.dot(qvec) * invDet;
                    if(t < 0) {
                        continue;
                    }
                    
                    // for first iteration
                    if(t < closest_t || closest_t < 0) {
                        normal = v0.cross(v1).normalize();
                        closest_t = v0v2.dot(qvec) * invDet;
                    }
                }
                return closest_t;
            }
            default:
                return -1;
        }
    }

    Vec3 get_normal(const Vec3 hit_point) const {
        switch (type)
        {
        case SPHERE:
        case PLANE:
        case MODEL:
            // TODO: implement normal for the model, based on the normal which the current triangle laid on
            // we don't have a way to know which triangle we hit

        default:
            return {0.f, 0.f, 0.f};
        }
    }
};

struct RaycastResult {
    Solid *hitted_object;
    Vec3 hit_point;
    Vec3 normal;
};

struct Light {
    Vec3 position;
    Vec3 color;
    float radius;
};

struct Scene {
    std::vector<Solid*> solids;

    std::vector<Light*> lights;

    bool hit(const Ray &ray, RaycastResult &result) const {
        float closest_t = INFINITY;
        Solid *closest_object = nullptr;
        Vec3 normal_of_closest_object; 
        for(auto solid: solids) {
            Vec3 normal;
            auto t = solid->hit(ray, normal);
            // avoid t too small (shadow acne)
            if(t <= EPSILON) {
                continue;
            }

            if(t < closest_t) {
                normal_of_closest_object = normal;
                closest_t = t;
                closest_object = solid;
            }
        }

        if(closest_object) {
            result.hitted_object = closest_object;
            result.hit_point = ray.at(closest_t);
            result.normal = normal_of_closest_object;
            return true;
        }
        return false;
    }
};