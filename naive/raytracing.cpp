#include <stdlib.h>
#include <stdio.h>
#include <cmath>
// for easy time measure
#include <ctime>

#include "./bitmap.h"
#include "./cmdline.h"
#include "./vec3.h"
#include "./parser.h"
#include "./scene.h"

float randf() {
    // random utility function
    // returns a random float from 0 to 1 excluded
    return ((float)rand()) / (float) RAND_MAX; 
}

Vec3 reflect(const Vec3 &axis, const Vec3 &vector) {
   return vector - axis * 2 * vector.dot(axis);
}

Vec3 get_bounce_direction(const Vec3 &normal, const Vec3 &looking_direction, MaterialType type) {
    switch (type)
    {
        case PLASTIC:
            return normal + Vec3::random();
        case METAL:
            return reflect(normal, looking_direction);
        default:
            printf("Error not handled material of type %i", type);
            return {0, 0, 0};
    }
}

// main recursive function
Vec3 cast(const Scene &scene, const Ray &ray, int k=10) {
    if(k <= 0) {
        // printf("best dark!!");
        return {0, 0, 0};
    }
        
    
    Vec3 color = {1, 1, 1};

    RaycastResult result;
    if(scene.hit(ray, result)) {
        auto hitted_object = result.hitted_object;
        Vec3 diffuse_color = {0, 0, 0};
        
        if(ray.direction.dot(result.normal) > 0.001) {
            // TODO in place operator
            // reverse the normal when we it a internal surface
            result.normal = result.normal * -1;
        }
        // DEBUG:
        // return (result.normal + Vec3{1.f, 1.f, 1.f}) * .5;

        // printf("calculate diffuse color for each light\n");
        for(auto light: scene.lights) {
            // check if some object occlude the light
            auto v = (light->position - result.hit_point).normalize();
            Ray light_ray = {result.hit_point, v};
            RaycastResult occlusion_raycast;
            auto distance_from_light = result.hit_point.distance(light->position);
            if(scene.hit(light_ray, occlusion_raycast)) {
                auto distance_from_occluder = occlusion_raycast.hit_point.distance(result.hit_point);
                if (distance_from_occluder <= distance_from_light) {
                    continue;
                } 
                // here the light is not really occluded, because the object is behind the light
            }

            // add to the diffusion component the diffusion light for this source
            auto diffuse_effect = v.dot(result.normal);
            if(diffuse_effect > 0.001) {
                auto d = __max(1, distance_from_light / light->radius);
                auto decay_rate = 1 / (d * d);
                diffuse_color = diffuse_color + light->color * decay_rate * diffuse_effect;
            }
        }

        // printf("make this ray bounce\n");
        auto material = hitted_object->material;
        auto bounce_direction = get_bounce_direction(result.normal, ray.direction, material.type);
        auto reflected = Ray{result.hit_point, bounce_direction};
        // printf("now recursive call, bounce direction: %f, %f, %f\n", bounce_direction.x, bounce_direction.y, bounce_direction.z);
        // printf("normal of hitted point: %f, %f, %f\n", result.normal.x, result.normal.y, result.normal.z);
        auto bounced_color = cast(scene, reflected, k - 1);
        // printf("----  after recursive call!\n");
        float diffusek = material.type == PLASTIC ? .7f : .2f,
            speculark = material.type == PLASTIC ? .2f : .9f;
        color = material.pigment * (diffuse_color * diffusek + bounced_color * speculark);
    }
    return color;
}

void to_hex_color(const Vec3 &color, char *r, char *g, char *b) {
    // converts a Vec3 float color to a triple int 0-255 rappresentation 
    *r = (int)(__min(color.x, 1) * 255.9);
    *g = (int)(__min(color.y, 1) * 255.9);
    *b = (int)(__min(color.z, 1) * 255.9);
}


int main(int argc, char *argv[]) {
    CommandLineOptions options;
    if(!parse_command_line(argc, argv, &options)) {
        return 1;
    }
    
    auto parser_result = parse_scene(options.scene_path);
    auto scene = parser_result.scene;
    int width = parser_result.width,
        height = parser_result.height;

    if (scene == nullptr) {
        printf("Error trying to parse the scene");
        return 1;
    }

    int image_size = width * height * 3;
    char* image_data = (char*) malloc(image_size * sizeof(char));
    
    float zoom = -1.0f;
    
    // getting the time at the start of the render
    auto start_time = clock();

    // render loop over each pixel of the image
    for(float x = 0; x < width; x++) {
        for(float y = 0; y < height; y++) {
            printf("done %f/100 done\r", (x * width + y) / (width * height) * 100);
            Vec3 color = {0, 0, 0}; 
            const int sample_per_pixel = options.sample_rate;
            for(int sample = 0; sample < sample_per_pixel; sample++) {
                float x_offset = randf() - .5,
                      y_offset = randf() - .5;
                // getting pixel ray coordinate
                float u = (x + x_offset - width / 2) / width,
                      v = (y + y_offset - height / 2) / height;
                Vec3 cam_position = {0.0f, 0.0f, 0.0f};
                Ray ray = {cam_position, Vec3{u, v, zoom} - cam_position};

                // hit with the scene
                // TODO: Vec3 += operator
                color = color + cast(*scene, ray) / sample_per_pixel;
            }

            // write vector color to the buffer
            char r, g, b;
            to_hex_color(color, &r, &g, &b);
            image_data[(int)(x * width + y) * 3]     = r;
            image_data[(int)(x * width + y) * 3 + 1] = g;
            image_data[(int)(x * width + y) * 3 + 2] = b;
        }
    }


    // save the image data to file
    auto outfile = options.outfile;
    Bitmap bmp = {width, height, image_data};
    save_bitmap(outfile, &bmp);

    // report to the user the overall render time
    auto end_time = clock();
    auto elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("rendered '%s' in %f seconds", outfile, elapsed_time);

    // free resources
    free(image_data);
    delete scene;

    return 0;
}