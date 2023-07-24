#pragma once

struct Bitmap {
    int width;
    int height;
    char* data;
};

bool save_bitmap(const char *path, const Bitmap *bitmap);
