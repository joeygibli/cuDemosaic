#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <libraw/libraw.h>
#include "image.h"
#include "ppm.h"
#include "CycleTimer.h"

typedef enum {RED, GREEN0, GREEN1, BLUE} color_t; // GREEN0 = Even Row

void print_vec(__m256i a) {
	short *thing = (short*)&a;
	for (int i = 0; i < 16; i++) {
		printf("%d ", thing[i]);
	}
	printf("\n");
}

inline color_t filter_color(int row, int col) {
    return (color_t) (((row & 1) << 1) + (col & 1));
}

// Define masks for blending vectors
// mask_op(a, b, mask) means if mask[i] == 1 then b[i:i+7] else a[i:i+7])
const __m256i full_mask = _mm256_setr_epi32(-1, -1, -1, -1, -1, -1, -1, -1);
const __m256i evens_mask = _mm256_setr_epi16(-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0);
const __m256i odds_mask = _mm256_setr_epi16(0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1);

/* 
 * interpolate_red_row processes a row of raw image data from a 
 * row containing red and green pixels. This infers the missing
 * red data on a green pixel and missing green data on a red pixel.
 */
inline __m256i interpolate_red_row(unsigned short *mem, int width) {
  __m256i avg_rg = _mm256_loadu_si256((__m256i*)mem);

  __m256i rg2 = _mm256_loadu_si256((__m256i*)(mem+2));
  __m256i above = _mm256_loadu_si256((__m256i*)(mem-width+1));
  __m256i below = _mm256_loadu_si256((__m256i*)(mem+width+1));

  avg_rg = _mm256_adds_epi16(avg_rg,rg2);

  __m256i avg_g = _mm256_adds_epi16(avg_rg, above);
  avg_g = _mm256_adds_epi16(avg_g, below);

  avg_g = _mm256_srli_epi16(avg_g, 2);
  avg_rg = _mm256_srli_epi16(avg_rg, 1);

  avg_rg = _mm256_blendv_epi8(avg_rg, avg_g, odds_mask);

  return avg_rg;
}

/* 
 * interpolate_blue_row processes a row of raw image data from a 
 * row containing blue and green pixels. This infers the missing
 * blue data on a green pixel and missing green data on a blue pixel.
 */
inline __m256i interpolate_blue_row(unsigned short *mem, int width) {
  __m256i avg_bg = _mm256_loadu_si256((__m256i*)mem);

  __m256i bg2 = _mm256_loadu_si256((__m256i*)(mem+2));
  __m256i above = _mm256_loadu_si256((__m256i*)(mem-width+1));
  __m256i below = _mm256_loadu_si256((__m256i*)(mem+width+1));

  avg_bg = _mm256_adds_epi16(avg_bg,bg2);

  __m256i avg_g = _mm256_adds_epi16(avg_bg,above);
  avg_g = _mm256_adds_epi16(avg_g,below);

  avg_g = _mm256_srli_epi16(avg_g,2);
  avg_bg = _mm256_srli_epi16(avg_bg,1);

  avg_bg = _mm256_blendv_epi8(avg_bg, avg_g, evens_mask);

  return avg_bg;
}


// Returns a mask to handle overflow so interpolation does not bleed 
// onto following row.
inline __m256i get_remainder(int width) {
  __m256i remainder;
    switch ((width-2)%16) {
      case 0: remainder = _mm256_setr_epi16(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
              break;
      case 1: remainder = _mm256_setr_epi16(-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
              break;
      case 2: remainder = _mm256_setr_epi16(-1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
              break;
      case 3: remainder = _mm256_setr_epi16(-1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
              break;
      case 4: remainder = _mm256_setr_epi16(-1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
              break;
      case 5: remainder = _mm256_setr_epi16(-1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
              break;
      case 6: remainder = _mm256_setr_epi16(-1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
              break;
      case 7: remainder = _mm256_setr_epi16(-1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
              break;
      case 8: remainder = _mm256_setr_epi16(-1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1);
              break;
      case 9: remainder = _mm256_setr_epi16(-1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1);
              break;
      case 10: remainder = _mm256_setr_epi16(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1);
              break;
      case 11: remainder = _mm256_setr_epi16(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1);
              break;
      case 12: remainder = _mm256_setr_epi16(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1);
              break;
      case 13: remainder = _mm256_setr_epi16(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1);
              break;
      case 14: remainder = _mm256_setr_epi16(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1);
              break;
      case 15: remainder = _mm256_setr_epi16(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1);
              break;
    }
    return remainder;
}

/*
 * write_image: given the raw Bayer image, horizontal, and veritcal interpolation
 *              writes the final image in LibRaw readable format in RGB order
 */
void write_image(unsigned short* img, unsigned short* bayer, 
  unsigned short* horz, unsigned short* vert, int width, int height) {
  #pragma omp parallel for collapse(2)
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      int px = i*width+j;
      switch (filter_color(i,j)) {
        case RED: 
          img[4*px] = bayer[px];
          img[4*px+1] = horz[px];
          img[4*px+2] = vert[px];
          break;
        case GREEN0: 
          img[4*px] = horz[px];
          img[4*px+1] = bayer[px];
          img[4*px+2] = vert[px];
          break;
        case GREEN1:
          img[4*px] = vert[px];
          img[4*px+1] = bayer[px];
          img[4*px+2] = horz[px];
          break;          
        case BLUE: 
          img[4*px] = vert[px];
          img[4*px+1] = horz[px];
          img[4*px+2] = bayer[px];
          break;
      }
    }
  }
}
/*
 * bilinear_avx: given an raw image input, outputs the bilinear demosaic
 *               in parallel using Intel Intrinsics and OMP paralellism.
 *               Per industry standard, excludes the outer border of the RAW 
 *               image due to insufficient color information.
 *
 * input: RAW Bayer Filter input image with varying color intensity levels
 * output_horz: horizontal interpolation output (infer color value on same row)
 * output_vert: vertical interpolation output (infer color value on row vert_upper)
 * width: image width
 * height: image height
 * nthreads: number of threads
 */
void bilinear_avx(unsigned short* input, unsigned short* output_horz, 
		  unsigned short* output_vert, int width, int height, int nthreads) {

// Run in parallel over columns of size 16 (width of AVX vector) excluding last column
#pragma omp parallel for schedule(static,1) num_threads(nthreads)
  for (int j = 0; j < width-17; j += 16) {
    // Initial row (assumed to start with BLUE and GREEN1 row)
    __m256i horz_above = _mm256_loadu_si256((__m256i*)(&input[j]));
    __m256i offset = _mm256_loadu_si256((__m256i*)(&input[j+2]));
    horz_above = _mm256_adds_epi16(horz_above, offset);
    __m256i vert_interp = horz_above;
    __m256i horz_interp = interpolate_blue_row(&input[j+width], width);

    for (int i = width; i < width*height-width; i += width) {
      if ((i/width)%2 == 0) { // Current row is a RED and GREEN0 row
        // Horizontal interpolation of row below (BLUE and GREEN1)
        __m256i horz_lower = interpolate_blue_row(&input[i+j+width], width);

        // Store the previous horizontal (horz_interp) and vertical (vert_interp) interpolation 
        _mm256_maskstore_epi32((int*)(&output_horz[i+j+1]), full_mask, horz_interp);
        _mm256_maskstore_epi32((int*)(&output_vert[i+j-width+1]), full_mask, vert_interp);

        // Sum the rows above and below to get the BLUE value for GREEN0 pixels
        vert_interp = _mm256_loadu_si256((__m256i*)(&input[i+j+width+1]));
        __m256i vert_upper = _mm256_loadu_si256((__m256i*)(&input[i+j-width+1]));
        vert_interp = _mm256_avg_epu16(vert_interp, vert_upper);

        // Sum the horizontal interpolations to get the BLUE value for RED pixels (diagonals)
        horz_above = _mm256_avg_epu16(horz_above, horz_lower);

        // Blend vectors to get approporiate BLUE values
        vert_interp = _mm256_blendv_epi8(vert_interp, horz_above, odds_mask);

        horz_above = horz_interp;
        horz_interp = horz_lower;

      } else { // Current row is BLUE and GREEN1 row
        // Horizontal interpolation of row below (RED and GREEN0)
        __m256i horz_lower = interpolate_red_row(&input[i+j+width], width);

        // Store the previous horizontal (horz_interp) and vertical (vert_interp) interpolation
        _mm256_maskstore_epi32((int*)(&output_horz[i+j+1]), full_mask, horz_interp);
        _mm256_maskstore_epi32((int*)(&output_vert[i+j-width+1]), full_mask, vert_interp);

        // Sum the rows above and below to get the RED value for GREEN1 pixels
        vert_interp = _mm256_loadu_si256((__m256i*)(&input[i+j+width+1]));
        __m256i vert_upper = _mm256_loadu_si256((__m256i*)(&input[i+j-width+1]));
        vert_interp = _mm256_avg_epu16(vert_interp, vert_upper);

        // Sum the horizontal interpolations to get the RED value for BLUE pixels (diagonals)
        horz_above = _mm256_avg_epu16(horz_above, horz_lower);

        // Blend vectors to get approporiate RED values
        vert_interp = _mm256_blendv_epi8(vert_interp, horz_above, evens_mask);

        horz_above = horz_interp;
        horz_interp = horz_lower;
      }
    }
    // Save image row's last vertical interpolation
    _mm256_maskstore_epi32((int*)(&output_vert[width*(height-2)+j+1]), full_mask, vert_interp);
  }

  
  // Perform interpolation on the final column using the remainder mask.
  // For performance reasons, this operation is separate to avoid forking.
  int j = width-16;
  __m256i remainder = get_remainder(width);

  __m256i horz_above = _mm256_loadu_si256((__m256i*)(&input[j]));
  __m256i offset = _mm256_loadu_si256((__m256i*)(&input[j+2]));
  horz_above = _mm256_adds_epi16(horz_above, offset);
  __m256i vert_interp = horz_above;
  __m256i horz_interp = interpolate_blue_row(&input[j+width], width);

  for (int i = width; i < width*height-width; i += width) {
    if ((i/width)%2 == 0) {
      __m256i horz_lower = interpolate_blue_row(&input[i+j+width], width);
      _mm256_maskstore_epi32((int*)(&output_horz[i+j+1]), remainder, horz_interp);
      _mm256_maskstore_epi32((int*)(&output_vert[i+j-width+1]), remainder, vert_interp);

      vert_interp = _mm256_loadu_si256((__m256i*)(&input[i+j+width+1]));
      __m256i vert_upper = _mm256_loadu_si256((__m256i*)(&input[i+j-width+1]));
      vert_interp = _mm256_avg_epu16(vert_interp, vert_upper);

      horz_above = _mm256_avg_epu16(horz_above, horz_lower);

      vert_interp = _mm256_blendv_epi8(vert_interp, horz_above, odds_mask);

      horz_above = horz_interp;
      horz_interp = horz_lower;

    } else {
      __m256i horz_lower = interpolate_red_row(&input[i+j+width], width);
      _mm256_maskstore_epi32((int*)(&output_horz[i+j+1]), remainder, horz_interp);
      _mm256_maskstore_epi32((int*)(&output_vert[i+j-width+1]), remainder, vert_interp);

      vert_interp = _mm256_loadu_si256((__m256i*)(&input[i+j+width+1]));
      __m256i vert_upper = _mm256_loadu_si256((__m256i*)(&input[i+j-width+1]));
      vert_interp = _mm256_avg_epu16(vert_interp, vert_upper);
      horz_above = _mm256_avg_epu16(horz_above, horz_lower);

      vert_interp = _mm256_blendv_epi8(vert_interp, horz_above, evens_mask);

      horz_above = horz_interp;
      horz_interp = horz_lower;
    }
    
  }
  _mm256_maskstore_epi32((int*)(&output_vert[width*(height-2)+j+1]), remainder, vert_interp);
}

/*
 * debug_corner: output contents of image and filter to debug content
 */
void debug_corner(unsigned short *img, unsigned short *bayer, unsigned short *horz, unsigned short *vert,
  int width, int height) {
   printf("Image data:\n");
    for (int i = 0; i < 12; i++) {
      for (int j = 0; j < 4*16; j++) {
        printf("%d ", img[(4*i*width)+j]);
      }
      printf("\n");
    }
    printf("\n");
    printf("Bayer data:\n");
    for (int i = 0; i < 12; i++) {
      for (int j = 0; j < 16; j++) {
        printf("%d ", bayer[i*width+j]);
      }
      printf("\n");
    }
    printf("\n");
    printf("Horz data:\n");
    for (int i = 0; i < 12; i++) {
      for (int j = 0; j < 16; j++) {
        printf("%d ", horz[i*width+j]);
      }
      printf("\n");
    }
    printf("\n");
    printf("Vert data:\n");
    for (int i = 0; i < 12; i++) {
      for (int j = 0; j < 16; j++) {
        printf("%d ", vert[i*width+j]);
      }
      printf("\n");
    }
    printf("\n");
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

    // Allocate interpolation arrays 
    unsigned short *output_horz = (unsigned short*)malloc(width*height*sizeof(short));
    unsigned short *output_vert = (unsigned short*)malloc(width*height*sizeof(short));

    for (int n_threads = 1; n_threads <= 8; n_threads*= 2) {
    double startTime = CycleTimer::currentSeconds();
    for (int i = 0; i < 20; i++) {
        bilinear_avx(iProcessor.imgdata.rawdata.raw_image, output_horz, output_vert, width, height, n_threads);
    }
    write_image(img->data, iProcessor.imgdata.rawdata.raw_image,
      output_horz, output_vert, width, height);
    double totalTime = CycleTimer::currentSeconds() - startTime;
    double avgTime = totalTime / 20.0;

    printf("AVX: avg time = %lf ms using %d threads\n", avgTime * 1000, n_threads);
    }
    free(output_horz);
    free(output_vert);

    img->clear(0, 0, 0, 0);

    iProcessor.recycle();
}
