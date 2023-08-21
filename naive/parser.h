#include "./scene.h"

struct ParserResult {
    int width, height;
    Scene *scene;
};

ParserResult parse_scene(const char *filename);