#include <libraw/libraw.h>

#include "image.h"
#include "ppm.h"
#include "CycleTimer.h"

#include "demosaic-cu.h"

typedef enum {RED, GREEN1, GREEN2, BLUE} color_t; // green1 = even row

inline color_t fc(int row, int col) {
    return (color_t) (((row & 1) << 1) + (col & 1));
}

inline int updiv(int n, int d) {
  return (n + d - 1)/d;
}

inline int min(int x, int y) {
  return x < y ? x : y;
}

void linear_demosaic_block_omp(ushort *raw_in, ushort *out, int width, int height) {
  const int block_size = 32;

#pragma omp parallel for collapse(2)
  for (int block_row = 0; block_row < updiv(height,block_size); block_row++) {
    for (int block_col = 0; block_col < updiv(width,block_size); block_col++) {
      int px_row = block_size*block_row;
      int px_col = block_size*block_col;
      int row_bound = min(px_row+block_size, height);
      int col_bound = min(px_col+block_size, width);
      //#pragma omp parallel for num_threads(2)
      for (int row = px_row; row < row_bound; row++) {
	for (int col = px_col; col < col_bound; col++) {
	  ushort r, g, b;
	  int idx = row * width + col;
	  switch (fc(row, col)) {
	  case RED:
	    r = raw_in[idx];
	    g = (raw_in[(row - 1) * width + col] +
		 raw_in[(row + 1) * width + col] +
		 raw_in[row * width + (col - 1)] +
		 raw_in[row * width + (col + 1)])
	      / 4;
	    b = (raw_in[(row - 1) * width + (col - 1)] +
		 raw_in[(row - 1) * width + (col + 1)] +
		 raw_in[(row + 1) * width + (col - 1)] +
		 raw_in[(row + 1) * width + (col + 1)])
	      / 4;
	    break;
	  case GREEN1:
	    r = (raw_in[row * width + (col - 1)] +
		 raw_in[row * width + (col + 1)])
	      / 2;
	    g = raw_in[idx];
	    b = (raw_in[(row + 1) * width + col] +
		 raw_in[(row - 1) * width + col])
	      / 2;
	    break;
	  case GREEN2:
	    r = (raw_in[(row + 1) * width + col] +
		 raw_in[(row - 1) * width + col])
	      / 2;
	    g = raw_in[idx];
	    b = (raw_in[row * width + (col - 1)] +
		 raw_in[row * width + (col + 1)])
	      / 2;
	    break;
	  case BLUE:
	    b = raw_in[idx];
	    g = (raw_in[(row - 1) * width + col] +
		 raw_in[(row + 1) * width + col] +
		 raw_in[row * width + (col - 1)] +
		 raw_in[row * width + (col + 1)])
	      / 4;
	    r = (raw_in[(row - 1) * width + (col - 1)] +
		 raw_in[(row - 1) * width + (col + 1)] +
		 raw_in[(row + 1) * width + (col - 1)] +
		 raw_in[(row + 1) * width + (col + 1)])
	      / 4;
	    break;
	  }
	  out[idx * 4] = r;
	  out[idx * 4 + 1] = g;
	  out[idx * 4 + 2] = b;
	}
      }
    }
  }
}

void linear_demosaic_omp(ushort *raw_in, ushort *out, int width, int height) {
    #pragma omp parallel for collapse(2)
    for (int row = 1; row < height - 1; row++) {
        for (int col = 1; col < width - 1; col++) {
            int idx = row * width + col;
            ushort r, g, b;
            switch (fc(row, col)) {
            case RED:
                r = raw_in[idx];
                g = (raw_in[(row - 1) * width + col] +
                     raw_in[(row + 1) * width + col] +
                     raw_in[row * width + (col - 1)] +
                     raw_in[row * width + (col + 1)])
                    / 4;
                b = (raw_in[(row - 1) * width + (col - 1)] +
                     raw_in[(row - 1) * width + (col + 1)] +
                     raw_in[(row + 1) * width + (col - 1)] +
                     raw_in[(row + 1) * width + (col + 1)])
                    / 4;
                break;
            case GREEN1:
                r = (raw_in[row * width + (col - 1)] +
                     raw_in[row * width + (col + 1)])
                    / 2;
                g = raw_in[idx];
                b = (raw_in[(row + 1) * width + col] +
                     raw_in[(row - 1) * width + col])
                    / 2;
                break;
            case GREEN2:
                r = (raw_in[(row + 1) * width + col] +
                     raw_in[(row - 1) * width + col])
                    / 2;
                g = raw_in[idx];
                b = (raw_in[row * width + (col - 1)] +
                     raw_in[row * width + (col + 1)])
                    / 2;
                break;
            case BLUE:
                b = raw_in[idx];
                g = (raw_in[(row - 1) * width + col] +
                     raw_in[(row + 1) * width + col] +
                     raw_in[row * width + (col - 1)] +
                     raw_in[row * width + (col + 1)])
                    / 4;
                r = (raw_in[(row - 1) * width + (col - 1)] +
                     raw_in[(row - 1) * width + (col + 1)] +
                     raw_in[(row + 1) * width + (col - 1)] +
                     raw_in[(row + 1) * width + (col + 1)])
                    / 4;
                break;
            }
            out[idx * 4] = r;
            out[idx * 4 + 1] = g;
            out[idx * 4 + 2] = b;
        }
    }
}

int main(int argc, char** argv) {
    // Let us create an image processor
    LibRaw iProcessor;
    
    // Open the file and read the metadata
    iProcessor.open_file(argv[1]);

    int width = iProcessor.imgdata.sizes.width;
    int height = iProcessor.imgdata.sizes.height;
    // The metadata are accessible through data fields of the class
    printf("Image size: %d x %d\n", width, height);
    
    // Let us unpack the image
    iProcessor.unpack();

    Image *img = new Image(width, height);

    double startTime = CycleTimer::currentSeconds();
    for (int i = 0; i < 20; i++) {
        linear_demosaic_omp(iProcessor.imgdata.rawdata.raw_image, img->data,
                            width, height);
    }
    double totalTime = CycleTimer::currentSeconds() - startTime;
    double avgTime = totalTime / 20.0;

    printf("OMP: avg time = %lf ms\n", avgTime * 1000);
    writePPMImage(img, "out-omp.ppm");

    img->clear(0, 0, 0, 0);
    
    /*    initStreams();
    startTime = CycleTimer::currentSeconds();
    for (int i = 0; i < 20; i++) {
        linear_demosaic_cu(iProcessor.imgdata.rawdata.raw_image, img->data,
                            width, height);
    }
    totalTime = CycleTimer::currentSeconds() - startTime;
    avgTime = totalTime / 20.0;

    printf("CUDA: avg time = %lf ms\n", avgTime * 1000);
    writePPMImage(img, "/tmp/out-cu.ppm");
    */
    
    // Finally, let us free the image processor for work with the next image
    iProcessor.recycle();
}
