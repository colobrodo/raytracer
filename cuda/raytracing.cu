
#include "stdio.h"
#include <iostream>
// for easy time measure
#include <ctime>

#include <curand_kernel.h>

#include "config.h"
#include "cuda_helper.h"
#include "bitmap.h"
#include "cmdline.h"
#include "vec3.h"
#include "parser.h"
#include "scene.h"

#define __max(a, b) ((a) >= (b) ? (a) : (b))
#define __min(a, b) ((a) <= (b) ? (a) : (b))


__host__ __device__ Vec3 reflect(const Vec3 &axis, const Vec3 &vector) {
   return vector - axis * 2 * vector.dot(axis);
}

__device__ Vec3 random_vector(curandState &random_state) {
    float x = curand_uniform(&random_state),
          y = curand_uniform(&random_state),
          z = curand_uniform(&random_state);
    return Vec3{x, y, z}.normalize();
}

__device__ Vec3 get_bounce_direction(const Vec3 &normal, const Vec3 &looking_direction, curandState &random_state, MaterialType type) {
    switch (type)
    {
        case PLASTIC:
            return normal + random_vector(random_state);
        case METAL:
            return reflect(normal, looking_direction);
        default:
            printf("Error not handled material of type %i", type);
            return {0, 0, 0};
    }
}

__host__ __device__ void to_hex_color(const Vec3 &color, char *r, char *g, char *b) {
    // converts a Vec3 float color to a triple int 0-255 rappresentation
    *r = (int)(__min(color.x, 1.f) * 255.9f);
    *g = (int)(__min(color.y, 1.f) * 255.9f);
    *b = (int)(__min(color.z, 1.f) * 255.9f);
}

__device__ void gamma_correction(Vec3 &color) {
    // https://www.cambridgeincolour.com/tutorials/gamma-correction.htm
    # define GAMMA 2.2
    color.x = powf(color.x, GAMMA);
    color.y = powf(color.y, GAMMA);
    color.z = powf(color.z, GAMMA);
}

__global__ void cast(Scene const *scene, curandState *rand_state, char *image_data, float width, float height, float offset) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = offset + threadIdx.y + blockIdx.y * blockDim.y;
    if(x >= width || y >= height) return;

    // initialize random state
    int pixel_index = y * width + x;
    curandState local_rand_state = rand_state[pixel_index];
    curand_init(SEED, pixel_index, 0, &local_rand_state);

    // camera variables (fixed at origin)
    Vec3 cam_position = {0.0f, 0.0f, 0.0f};
    float zoom = -1.f;

    Vec3 color = {0.f, 0.f, 0.f};

    for(int sample_i = 0; sample_i < N_SAMPLES; sample_i++) {
      Vec3 sampled_color = {0.f, 0.f, 0.f};
      // add random offset in the range of the pixel for oversampling
      float x_offset = curand_uniform(&local_rand_state) - .5f,
            y_offset = curand_uniform(&local_rand_state) - .5f;
      // getting pixel ray coordinate

      float u = (x + x_offset - width / 2) / width,
            v = (y + y_offset - height / 2) / height;
      Ray ray = {cam_position, Vec3{u, v, zoom} - cam_position};

      Ray current_ray = ray;
      float attenuation = 1.0f;
      for(int k = MAX_BOUNCE_DEPTH; k > 0; k--) {
          RaycastResult result;
          if(scene->hit(current_ray, result)) {
              auto hitted_object = result.hitted_object;
              Vec3 diffuse_color = {0.f, 0.f, 0.f};

              if(ray.direction.dot(result.normal) > 0.001) {
                  // TODO in place operator
                  // reverse the normal when we it a internal surface
                  result.normal = result.normal * -1;
              }

              for(int i = 0; i < scene->n_lights; i++) {
                  auto light = scene->lights[i];
                  // check if some object occlude the light
                  auto v = (light.position - result.hit_point).normalize();
                  Ray light_ray = {result.hit_point, v};
                  RaycastResult occlusion_raycast;
                  auto distance_from_light = result.hit_point.distance(light.position);
                  if(scene->hit(light_ray, occlusion_raycast)) {
                      auto distance_from_occluder = occlusion_raycast.hit_point.distance(result.hit_point);
                      if (distance_from_occluder <= distance_from_light) {
                          continue;
                      }
                      // here the light is not really occluded, because the object is behind the light
                  }

                  // add to the diffusion component the diffusion light for this source
                  auto diffuse_effect = v.dot(result.normal);
                  if(diffuse_effect > 0.001) {
                      auto d = __max(1, distance_from_light / light.radius);
                      auto decay_rate = 1 / (d * d);
                      diffuse_color = diffuse_color + light.color * decay_rate * diffuse_effect;
                  }
              }

              auto material = hitted_object->material;
              float diffusek = material.type == PLASTIC ? .8f : .1f,
                  speculark = material.type == PLASTIC  ? .2f : .9f;
              auto bounce_direction = get_bounce_direction(result.normal, current_ray.direction, local_rand_state, material.type);
              sampled_color = sampled_color + material.pigment * attenuation * (diffuse_color * diffusek);
              // update ray and specular component for the next iteration
              current_ray = Ray{result.hit_point, bounce_direction};
              // attenuate the next bounced ray by the current specular component
              attenuation *= speculark;
          } else {
              // we don't hit anything, the ray didn't bounce from anywhere so it's pure light
              sampled_color = sampled_color + Vec3{1.f, 1.f, 1.f} * attenuation;
              break;
          }
      }
      color = color + sampled_color / N_SAMPLES;
    }

    if(USE_GAMMA_CORRECTION) gamma_correction(color);

    char r, g, b;
    to_hex_color(color, &r, &g, &b);
    image_data[(int)(y * width + x) * 3]     = r;
    image_data[(int)(y * width + x) * 3 + 1] = g;
    image_data[(int)(y * width + x) * 3 + 2] = b;
}

int main(int argc, char *argv[]) {
    CommandLineOptions options;
    if(!parse_command_line(argc, argv, &options)) {
        printf("failed parsing command line options");
        return 1;
    }

    // parse scene from file
    auto parser_result = parse_scene(options.scene_path);
    Scene *scene = parser_result.scene;
    if (scene == nullptr) {
        printf("Error trying to parse the scene");
        return 1;
    }

    int width = parser_result.width,
        height = parser_result.height;

    // getting the time at the start of the render
    auto start_time = clock();

    int n_pixels = width * height * 3;
    int image_size = n_pixels * sizeof(char);
    // allocate memory for the image on the host
    // char *image_data = (char*) malloc(image_size);
    char *image_data;
    checkCudaErrors(cudaMallocHost((void**) &image_data, image_size));

    // allocate image data in device memory
    char *d_image_data;
    checkCudaErrors(cudaMalloc((void**) &d_image_data, image_size));

    // allocate random states, one curandState for each thread
    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **) &d_rand_state, width * height * sizeof(curandState)));

    // enforce use of L1 cache, instead of smem since we don't use the last one
    // checkCudaErrors(cudaFuncSetCacheConfig(cast, cudaFuncCachePreferL1));

    // create and initialize stop event for stream synchronization
    cudaEvent_t stop_event;
    checkCudaErrors(cudaEventCreate(&stop_event));

    // create the streams to operate
    cudaStream_t streams[N_STREAMS];
    for (int i = 0; i < N_STREAMS; i++) {
        checkCudaErrors(cudaStreamCreate(&streams[i]));
    }

    // the image is divide in horizontal strip where each strip is populated by
    // one different stream
    int strip_size = height / N_STREAMS;
    auto bytes_per_strip = width * strip_size * 3 * sizeof(char);

    // define grid and block sizes
    dim3 blocks(width / THREAD_SIZE_X + 1, strip_size / THREAD_SIZE_Y + 1);
    dim3 threads(THREAD_SIZE_X, THREAD_SIZE_Y);

    for (int i = 0; i < N_STREAMS; i++) {
        int offset = i * strip_size;
        cast<<<blocks, threads, 0, streams[i]>>>(scene, d_rand_state, d_image_data, width, height, offset);
    }

    for (int i = 0; i < N_STREAMS; i++) {
        int offset = i * strip_size;
        // Copy output (results) from GPU buffer to host memory.
        auto bytes_offset = i * bytes_per_strip;
        checkCudaErrors(cudaMemcpyAsync(&image_data[bytes_offset], &d_image_data[bytes_offset], bytes_per_strip, cudaMemcpyDeviceToHost, streams[i]));
    }

    // synchronize all streams on stop event
    checkCudaErrors(cudaEventRecord(stop_event, 0));
    checkCudaErrors(cudaEventSynchronize(stop_event));

    for (int i = 0; i < N_STREAMS; i++) {
        checkCudaErrors(cudaStreamDestroy(streams[i]));
    }

    // the reported time do not include saving images and dealing with io
    auto end_time = clock();

    // save the image data to file
    auto outfile = options.outfile;
    Bitmap bmp = {width, height, image_data};
    save_bitmap(outfile, &bmp);

    // report to the user the overall render time
    auto elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("rendered '%s' in %f seconds", outfile, elapsed_time);

    // free resources
    // free(image_data);
    // free cuda resources
    checkCudaErrors(cudaFree(d_image_data));
    checkCudaErrors(cudaFree(scene));
    // free random state
    checkCudaErrors(cudaFree(d_rand_state));

    return 0;
}
