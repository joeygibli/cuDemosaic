#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>

#include "image.h"



// writePPMImage --
//
// assumes input pixels are float4
// write 3-channel (8 bit --> 24 bits per pixel) ppm
void
writePPMImage(const Image* image, const char *filename)
{
    FILE *fp = fopen(filename, "wb");

    if (!fp) {
        fprintf(stderr, "Error: could not open %s for write\n", filename);
        exit(1);
    }

    // write ppm header
    fprintf(fp, "P6\n");
    fprintf(fp, "%d %d\n", image->width, image->height);
    fprintf(fp, "255\n");

    for (int j=image->height-1; j>=0; j--) {
        for (int i=0; i<image->width; i++) {

            const unsigned short* ptr = &image->data[4 * (j*image->width + i)];

            char val[3];
            val[0] = static_cast<char>(ptr[0] / 16);
            val[1] = static_cast<char>(ptr[1] / 16);
            val[2] = static_cast<char>(ptr[2] / 16);

            fputc(val[0], fp);
            fputc(val[1], fp);
            fputc(val[2], fp);
        }
    }

    fclose(fp);
    printf("Wrote image file %s\n", filename);
}
