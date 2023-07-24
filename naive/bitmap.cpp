#include <stdio.h>
#include "./bitmap.h"

bool save_bitmap(const char *path, const Bitmap *bitmap) {
    const int header_size = 54;

    int data_size = bitmap->height * bitmap->width * 3,
        file_size = header_size +  data_size;
    unsigned char header[14] = {'B','M', 0,0,0,0, 0,0, 0,0, 54,0,0,0};
    unsigned char info_header[40] = {40,0,0,0, 0,0,0,0, 0,0,0,0, 1,0, 24,0};
    unsigned char bmppad[3] = {0,0,0};

    header[ 2] = (unsigned char)(file_size    );
    header[ 3] = (unsigned char)(file_size>> 8);
    header[ 4] = (unsigned char)(file_size>>16);
    header[ 5] = (unsigned char)(file_size>>24);

    info_header[ 4] = (unsigned char)(bitmap->width      );
    info_header[ 5] = (unsigned char)(bitmap->width >>  8);
    info_header[ 6] = (unsigned char)(bitmap->width >> 16);
    info_header[ 7] = (unsigned char)(bitmap->width >> 24);
    info_header[ 8] = (unsigned char)(bitmap->height      );
    info_header[ 9] = (unsigned char)(bitmap->height >>  8);
    info_header[10] = (unsigned char)(bitmap->height >> 16);
    info_header[11] = (unsigned char)(bitmap->height >> 24);
    
    auto file = fopen(path, "wb");
    if(!file) {
        return false;
    }

    fwrite(header, 1, 14, file);
    fwrite(info_header, 1, 40, file);

    // fputs(bitmap->data, file);

    // TODO: refactor
    for(int i = 0; i < bitmap->height; i++) {
        for(int j = 0; j < bitmap->width; j++) {
            int y = bitmap->height - j - 1,
                pixel_index = (i * bitmap->width + j) * 3;
            // blue
            fwrite(&bitmap->data[pixel_index + 2], 1, 1, file);
            // green
            fwrite(&bitmap->data[pixel_index + 1], 1, 1, file);
            // red 
            fwrite(&bitmap->data[pixel_index + 0], 1, 1, file);
        }
    }


    fclose(file);
    return true;
}