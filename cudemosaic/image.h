#ifndef  __IMAGE_H__
#define  __IMAGE_H__


struct Image {

    Image(int w, int h) {
        width = w;
        height = h;
        data = new unsigned short[4 * width * height];
    }

    void clear(unsigned short r, unsigned short g, unsigned short b, unsigned short a) {

        int numPixels = width * height;
        unsigned short* ptr = data;
        for (int i=0; i<numPixels; i++) {
            ptr[0] = r;
            ptr[1] = g;
            ptr[2] = b;
            ptr[3] = a;
            ptr += 4;
        }
    }

    int width;
    int height;
    unsigned short* data;
};


#endif
