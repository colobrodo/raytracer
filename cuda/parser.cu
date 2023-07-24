#include <iostream>
#include <fstream>
#include <string.h>

#include "./scene.h"
#include "./cuda_helper.h"

/*
simple format for describing a scene with basic geometries,
here the following grammar:
p

vec3 := '(' float ',' float ',' float ')'

predefined_color := 'white' | 'black' | 'red' | 'blue' | 'cyan' | 'yellow' | 'green' | 'fuchsia'
color := vec3 | predefined_color

material_type := 'plastic' | 'metal'
material := (material_type ':')? color

sphere := 'sphere' vec3 float material
light := 'light' vec3 color
model := 'model' string

object_declaration := sphere | light

size_decl := 'size' float float
header := size

file := header object_declaration*
*/

struct ParserResult {
    std::vector<Solid> solids;
    std::vector<Light> lights;
};

Scene *make_scene(std::vector<Solid> solids, std::vector<Light> lights) {
    // allocate scene solids and lights on the unified memory
    auto scene_size = sizeof(Scene);
    auto solids_size = solids.size() * sizeof(Solid);
    auto lights_size = lights.size() * sizeof(Light);
    auto total_size = scene_size + solids_size + lights_size;

    Scene *scene;
    checkCudaErrors(cudaMallocManaged((void**) &scene, total_size));
    *scene = Scene{};
    scene->solids = (Solid*) ((void*) scene) + scene_size;
    for(int i = 0; i < solids.size(); i++) {
        scene->solids[i] = solids[i];
    }
    scene->n_solids = solids.size();

    scene->lights = (Light*) ((void*) scene) + scene_size + solids_size;
    for(int i = 0; i < lights.size(); i++) {
        scene->lights[i] = lights[i];
    }
    scene->n_lights = lights.size();

    return scene;
}

struct SceneParser {
    std::istream &stream;
    int line   = 1;
    int column = 0;
    // each time we look peek a new token we consume the character from the string,
    // so if the user would look ahead (without consuming the token)
    // we save the just-seen token that is no more available from the stream
    std::string buffer = "";

    SceneParser(std::istream &input_stream): stream(input_stream) {}

    void eat_spaces() {
        // consume all the empty lines, spaces and comments before the next token
        while (true)
        {
            auto current_char = stream.peek();
            // comments
            if(current_char == '#') {
                // consume the characters until the end of the line
                // note: we don't consume the end-of-line here but at the end of the loop
                while(stream.peek() != '\n') advance();
            }
            else if (!iswspace(current_char)) {
                break;
            }
            advance();
        }
    }

    void advance() {
        auto current_char = stream.peek();
        if (current_char == '\n') {
            line += 1;
            column = 0;
        }
        else {
            column += 1;
        }
        stream.ignore();
    }


    float parse_float() {
        auto next_token = pop();
        char *error;
        float result = strtof(next_token.c_str(), &error);
        if (*error != '\0') {
            printf("error parsing file: cannot interp '%s' as a float at %i:%i", next_token.c_str(), line, column);
        }
        return result;
    }

    std::string parse_string() {
        auto next_token = pop();
        // remove quotes
        next_token = next_token.substr(1, next_token.size() - 2);
        return next_token;
    }

    std::string pop() {
        // check if we already peeked without eating the next token
        if(!buffer.empty()) {
            auto result = buffer;
            buffer = "";
            return result;
        }

        eat_spaces();
        std::string result;
        char current_char = stream.peek();
        // add the current char to the result string and advance
        auto enqueque = [&] () {
            result.append(1, current_char);
            advance();
            current_char = stream.peek();
        };

        switch (current_char)
        {
            // if char is a symbol return it
            case ',':
            case '(':
            case ')':
            case ':':
                advance();
                result.append(1, current_char);
                break;
            case '"': {
                    enqueque();
                    // do not handle escape for now
                    bool in_string = true;
                    while(true) {
                        enqueque();
                        // eat also the last quote of the string
                        if(!in_string) break;
                        in_string = current_char != '"';
                    }
                }
                break;
            // float parsing
            case '-':
            case '+':
            case '.':
            case '0': case '1': case '2': case '3': case '4': case '5': case '6': case '7': case '8': case '9':
                if(current_char == '+' || current_char == '-') {
                    enqueque();
                }

                while(isdigit(current_char)) {
                    enqueque();
                }
                if(current_char == '.') {
                    enqueque();
                    // TODO: here we should check if we already have encountred a digit before the dot
                    //       cause '.' is a valid float for now
                    while(isdigit(current_char)) enqueque();
                }
                break;

            default:
                while(isalpha(current_char)) {
                    enqueque();
                }
        }

        // clearing buffer
        buffer = "";

        return result;
    }

    std::string peek() {
        // peek always look ahead and save the result to the buffer
        if(buffer.empty()) {
            buffer = pop();
        }
        return buffer;
    }

    void match(char *expected_lexem) {
        // match primitive: consume a lexem from the list and if is different
        // from the expected one raise an error
        auto next_lexem = pop();
        if(next_lexem != expected_lexem) {
            printf("error parsing the scene file: expected '%s', getting '%s' instead at %i:%i\n", expected_lexem, next_lexem.c_str(), line, column);
            // _ASSERT(false);
        }
    }

    bool maybe_match(char *expected_lexem) {
        // variant of match that can fail
        // if the expected lexem is the next in the stream, we consume it and returns true.
        // return false otherwise leaving the stream untouched
        auto next_lexem = peek();
        if(next_lexem == expected_lexem) {
            pop();
            return true;
        }
        return false;
    }


    void parse_header() {
        // TODO: unused for now
        match("size");
        auto width  = parse_float();
        auto height = parse_float();
    }

    Vec3 parse_vec3() {
        match("(");
        auto x = parse_float();
        match(",");
        auto y = parse_float();
        match(",");
        auto z = parse_float();
        match(")");
        return {x, y, z};
    }

    Vec3 parse_color() {
        // predefined color
        if(maybe_match("red")) {
            return {1.f, 0.f, 0.f};
        } else if(maybe_match("blue")) {
            return {0.f, 0.f, 1.f};
        } else if(maybe_match("green")) {
            return {0.f, 1.f, 0.f};
        } else if(maybe_match("white")) {
            return {1.f, 1.f, 1.f};
        } else if(maybe_match("black")) {
            return {0.f, 0.f, 0.f};
        } else if(maybe_match("cyan")) {
            return {0.f, 1.f, 01.f};
        } else if(maybe_match("violet")) {
            return {1.f, 0.f, 1.f};
        } else if(maybe_match("fuchsia")) {
            return {0.96f, 0.f, 96.f};
        } else if(maybe_match("yellow")) {
            return {1.f, 1.f, 0.f};
        } else if(maybe_match("orange")) {
            return {.98f, .45f, .02f};
        }

        auto color = parse_vec3();
        return color;
    }

    Solid parse_sphere() {
        match("sphere");
        auto position = parse_vec3();
        auto radius = parse_float();
        // parse material
        MaterialType material_type = PLASTIC;
        if(maybe_match("metal")) {
            material_type = METAL;
            match(":");
        } else if(maybe_match("plastic")) {
            material_type = PLASTIC; // redundant but for clarity
            match(":");
        }
        auto color = parse_color();
        Material material = {color, material_type};
        return Solid{SPHERE, Sphere{position, radius}, material};
    }

    Solid parse_plane() {
        match("plane");
        auto position = parse_vec3();
        auto normal = parse_vec3();
        // parse material
        MaterialType material_type = PLASTIC;
        if(maybe_match("metal")) {
            material_type = METAL;
            match(":");
        } else if(maybe_match("plastic")) {
            material_type = PLASTIC; // redundant but for clarity
            match(":");
        }
        auto color = parse_color();
        Material material = {color, material_type};
        auto solid = Solid{PLANE};
        solid.plane = Plane{position, normal};
        solid.material = material;
        return solid;
    }

    Light parse_light() {
        match("light");
        auto position = parse_vec3();
        auto color = parse_color();
        return Light{position, color, 2.f};
    }

    Scene *parse_scene() {
        // main routine that parse the whole file
        parse_header();

        ParserResult result;
        while(true) {
            auto next_token = peek();

            if(next_token == "light") {
                auto light = parse_light();
                result.lights.push_back(light);
            } else if(next_token == "sphere") {
                auto sphere = parse_sphere();
                result.solids.push_back(sphere);
            } else if(next_token == "plane") {
                auto plane = parse_plane();
                result.solids.push_back(plane);
            /*
            } else if(next_token == "model") {
                auto model = parse_model();
                if(!model) {
                    continue;
                }
                scene->solids.push_back(model);
            */
            } else {
                // printf("encountred token, terminated parsing\n");
                break;
            }
        }
        return make_scene(result.solids, result.lights);
    }
};


Scene *parse_scene(const char *filename) {
    std::ifstream input_stream(filename);
    SceneParser parser(input_stream);
    auto scene = parser.parse_scene();
    input_stream.close();
    return scene;
}
