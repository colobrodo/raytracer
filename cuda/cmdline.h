#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <string.h>


struct CommandLineOptions {
    char *scene_path;
    char *outfile;
    int sample_rate = 20;
};

bool parse_command_line(int argc, char *argv[], CommandLineOptions *options) {
    auto print_error = [](char *error) {
        printf("Error: the command line options should be in this format\n");
        printf("\traytracing.exe 'path to scene' 'out path to bitmap' [--sample-rate rate]\n");
        printf(error);
    };

    if(argc < 3) {
        // TODO: add help string

        // scene path and outfile for bitmap are mandatory options
        print_error("You have not provide enough arguments, 'path to scene' and 'out path to bitmap' are mandatory!\n");

        return false;
    }
    options->scene_path = argv[1];
    options->outfile    = argv[2];
    // read all the optionals comands of type "--option-name value"
    int argument_index = 3;
    auto read_int = [&](int *value) {
        argument_index += 1;
        if(argument_index >= argc) {
            return false;
        }
        auto argument = argv[argument_index];
        *value = atoi(argument);
        return true;
    };

    for(;argument_index < argc; argument_index++) {
        auto option = argv[argument_index];
        /*
        if(strcmp(option, "--sample-rate") == 0) {
            if(!read_int(&options->sample_rate)) {
                print_error("Missing value for argument \"--sample-rate\" needed an integer\n");
                return false;
            }
        } else
        */
        {
            print_error("");
            printf("Unknow argument option \"%s\"\n", option);
            return false;
        }
    }

    return true;
}
