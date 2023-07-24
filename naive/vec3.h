#pragma once

#include <cmath>


union Vec3 {
    struct {
        float x, y, z;
    };
    float xyz[3];

    inline Vec3 operator+(const Vec3 &other) const {
        return {x + other.x, y + other.y, z + other.z};
    }

    inline Vec3 operator-(const Vec3 &other) const {
        return {x - other.x, y - other.y, z - other.z};
    }
    
    inline Vec3 operator-() const {
        return {-x, -y, -z};
    }

    inline Vec3 operator*(const Vec3 &other) const {
        return {x * other.x, y * other.y, z * other.z};
    }

    inline Vec3 operator*(const float factor) const {
        return {x * factor, y * factor, z * factor};
    }

    inline Vec3 operator/(const Vec3 &other) const {
        return {x / other.x, y / other.y, z / other.z};
    }

    inline Vec3 operator/(const float factor) const {
        auto k = 1 / factor;
        return {x * k, y * k, z * k};
    }

    inline Vec3 normalize() const {
        return *this / length();
    }
    inline Vec3 cross(const Vec3 &other) const {
        return {
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x,
        };
    }

    inline float distance(const Vec3 &other) const {
        float dx = x - other.x,
            dy = y - other.y,
            dz = z - other.z;
        return sqrt(dx * dx + dy * dy + dz * dz);
    }

    inline float dot(const Vec3 &other) const {
        return x * other.x + y * other.y + z * other.z;
    }
    
    float length() const {
        return sqrt(x * x + y * y + z * z);
    }

    static Vec3 random() {
        float x = ((float)rand()) / (float) RAND_MAX,
            y = ((float)rand()) / (float) RAND_MAX,
            z = ((float)rand()) / (float) RAND_MAX;
        return Vec3{x, y, z}.normalize();
    }
};
