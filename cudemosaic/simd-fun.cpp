#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>

void print_vec(__m256i a) {
	short *thing = (short*)&a;
	for (int i = 0; i < 16; i++) {
		printf("%d ", thing[i]);
	}
	printf("\n");
}

const __m256i full_mask = _mm256_setr_epi32(-1, -1, -1, -1, -1, -1, -1, -1);

__m256i interpolate_red_row(short *mem, int width) {
  __m256i avg_rg = _mm256_load_si256((__m256i*)mem);

  __m256i rg2 = _mm256_load_si256((__m256i*)(mem+2));
  __m256i above = _mm256_load_si256((__m256i*)(mem-width+1));
  __m256i below = _mm256_load_si256((__m256i*)(mem+width+1));

  avg_rg = _mm256_add_epi16(avg_rg,rg2);

  __m256i avg_g = _mm256_add_epi16(avg_rg,above);
  avg_g = _mm256_add_epi16(avg_g,below);

  // avg_g = _mm256_srai_epi16(avg_g,2);
  // avg_rg = _mm256_srai_epi16(avg_rg,1);

  avg_rg = _mm256_blend_epi16(avg_rg,avg_g,-21846);

  return avg_rg;
}

__m256i interpolate_blue_row(short *mem, int width) {
  __m256i avg_bg = _mm256_load_si256((__m256i*)mem);

  __m256i bg2 = _mm256_load_si256((__m256i*)(mem+2));
  __m256i above = _mm256_load_si256((__m256i*)(mem-width+1));
  __m256i below = _mm256_load_si256((__m256i*)(mem+width+1));

  avg_bg = _mm256_add_epi16(avg_bg,bg2);

  __m256i avg_g = _mm256_add_epi16(avg_bg,above);
  avg_g = _mm256_add_epi16(avg_g,below);

  // avg_g = _mm256_srai_epi16(avg_g,2);
  // avg_bg = _mm256_srai_epi16(avg_bg,1);

  avg_bg = _mm256_blend_epi16(avg_bg,avg_g,21845);

  return avg_bg;
}

__m256i get_remainder(int width) {
  __m256i remainder;
    switch ((width-2)%16){
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

void bilinear_avx(short* input, short* output_horz, short* output_vert, int width, int height) {
    __m256i remainder = get_remainder(width);

    for (int j = 0; j < width-1; j += 16) {
    __m256i within_above = _mm256_load_si256((__m256i*)(&input[j]));
    __m256i offset = _mm256_load_si256((__m256i*)(&input[j+2]));
    within_above = _mm256_add_epi16(within_above, offset);
    __m256i between = within_above;
    __m256i within = interpolate_red_row(&input[j+width], width);

    for (int i = width; i < width*height-width; i += width) {
      if ((i/width)%2 == 1) {
        __m256i within_below = interpolate_blue_row(&input[i+j+width], width);
        if (j+16 >= width-1) {
          _mm256_maskstore_epi32((int*)(&output_horz[i+j+1]), remainder, within);
          _mm256_maskstore_epi32((int*)(&output_vert[i+j-width+1]), remainder, between);
        } else {
          _mm256_maskstore_epi32((int*)(&output_horz[i+j+1]), full_mask, within);
          _mm256_maskstore_epi32((int*)(&output_vert[i+j-width+1]), full_mask, between);
        }

        between = _mm256_load_si256((__m256i*)(&input[i+j+width+1]));
        __m256i above = _mm256_load_si256((__m256i*)(&input[i+j-width+1]));
        between = _mm256_add_epi16(between, above);
        within_above = _mm256_add_epi16(within_above, within_below);

        between = _mm256_blend_epi16(between, within_above, 0b1010101010101010);

        within_above = within;
        within = within_below;
      } else {
        __m256i within_below = interpolate_red_row(&input[i+j+width], width);
        if (j+16 >= width-1) {
          _mm256_maskstore_epi32((int*)(&output_horz[i+j+1]), remainder, within);
          _mm256_maskstore_epi32((int*)(&output_vert[i+j-width+1]), remainder, between);
        } else {
          _mm256_maskstore_epi32((int*)(&output_horz[i+j+1]), full_mask, within);
          _mm256_maskstore_epi32((int*)(&output_vert[i+j-width+1]), full_mask, between);
        }

        between = _mm256_load_si256((__m256i*)(&input[i+j+width+1]));
        __m256i above = _mm256_load_si256((__m256i*)(&input[i+j-width+1]));
        between = _mm256_add_epi16(between, above);

        within_above = _mm256_add_epi16(within_above, within_below);

        between = _mm256_blend_epi16(within_above, between, 0b1010101010101010);

        within_above = within;
        within = within_below;
      }
      
    }

    if (j+16 >= width-1) {
        //_mm256_maskstore_epi32((int*)(&stor_arr[i+j-width+1]), remainder, within_above);
        _mm256_maskstore_epi32((int*)(&output_vert[width*(height-2)+j+1]), remainder, between);
      } else {
        //_mm256_maskstore_epi32((int*)(&stor_arr[i+j-width+1]), full_mask, within_above);
        _mm256_maskstore_epi32((int*)(&output_vert[width*(height-2)+j+1]), full_mask, between);
    }
  }
}

// __m256i interpolate_blue_row(short *mem, int width) {
//   __m256i avg_bg = _mm256_maskload_epi16(mem, full_mask);
//   __m256i bg2 = _mm256_maskload_epi32(mem+2, full_mask);
//   __m256i above = _mm256_maskload_epi32(mem-width+1, full_mask);
//   __m256i below = _mm256_maskload_epi32(mem+width+1, full_mask);

//   avg_bg = _mm256_add_epi32(avg_bg,bg2);

//   __m256i avg_g = _mm256_add_epi32(avg_bg,above);
//   avg_g = _mm256_add_epi32(avg_g,below);

//   avg_g = _mm256_srai_epi32(avg_g,2);
//   avg_bg = _mm256_srai_epi32(avg_bg,1);

//   avg_bg = _mm256_blend_epi32(avg_bg,avg_g,85);

//   return avg_bg;
// }

int main() {

  const int width = 40;
  const int height = 15;

  short load_arr[width*height];
  short stor_arr[width*height] = { 0 };
  short vert_arr[width*height] = { 0 };

  printf("Store Array:\n");
  for (int i = 0; i < height; i++) {
  	for (int j = 0; j < width; j++) {
  		printf("%d ", stor_arr[i*width+j]);
  	}
  	printf("\n");
  }
  printf("\n");

  for (int i = 0; i < height; i++) {
  	for (int j = 0; j < width; j++) {
      load_arr[i*width + j] = (i*width + j) % 9;
  		// if (i % 2 == 1)
  		//  	load_arr[i*width + j] = j % 2 == 0 ? 1 : 2;
  		// else
  		//  	load_arr[i*width + j] = j % 2 == 0 ? 2 : 3;
  	}
  }

  printf("Load Array:\n"); 
  for (int i = 0; i < height; i++) {
  	for (int j = 0; j < width; j++) {
  		printf("%d ", load_arr[i*width+j]);
  	}
  	printf("\n");
  }
  printf("\n");

  /* Initialize the mask vector */

  /* Selectively load data into the vector */
  // for (int i = 2*width+width; i < width*height-width; i += 2*width) {
  // 	for (int j = 0; j < width-1; j += 16) {
  //     __m256i avg_bg_up = interpolate_blue_row(&load_arr[i+j-width], width);
  // 		__m256i avg_rg = interpolate_red_row(&load_arr[i+j], width);
  //     __m256i avg_bg_down = interpolate_blue_row(&load_arr[i+j+width], width);
  //     __m256i avg_bg = _mm256_add_epi16(avg_bg_up, avg_bg_down);
  //     avg_bg = _mm256_srai_epi16(avg_bg, 1);

  //     __m256i b_avg = _mm256_load_si256((__m256i*)&load_arr[i+j+width+1]);
  //     __m256i b_up = _mm256_load_si256((__m256i*)&load_arr[i+j-width+1]);
  //     b_avg = _mm256_avg_epu16(b_avg, b_up);

  //     b_avg = _mm256_blend_epi16(b_avg, avg_bg, -21846);

  //     if (j+16 >= width-1) {
  //       _mm256_maskstore_epi32((int*)(&stor_arr[i+j+1]), remainder, avg_rg);
  //       _mm256_maskstore_epi32((int*)(&stor_arr[i+j-width+1]), remainder, avg_bg_up);
  //       _mm256_maskstore_epi32((int*)(&stor_arr[i+j+width+1]), remainder, avg_bg_down);
  //       _mm256_maskstore_epi32((int*)(&vert_arr[i+j+1]), remainder, b_avg);
  //     } else {
  //       _mm256_maskstore_epi32((int*)(&stor_arr[i+j+1]), full_mask, avg_rg);
  //       _mm256_maskstore_epi32((int*)(&stor_arr[i+j-width+1]), full_mask, avg_bg_up);
  //       _mm256_maskstore_epi32((int*)(&stor_arr[i+j+width+1]), full_mask, avg_bg_down);
  //       _mm256_maskstore_epi32((int*)(&vert_arr[i+j+1]), full_mask, b_avg);
  //     }
  // 		//printf("(%d, %d): ", i, j);
  // 		//print_vec(total);
  // 	}
  // }

  // for (int j = 0; j < width-1; j += 16) {
  //   __m256i within_above = _mm256_load_si256((__m256i*)(&load_arr[j]));
  //   __m256i offset = _mm256_load_si256((__m256i*)(&load_arr[j+2]));
  //   within_above = _mm256_add_epi16(within_above, offset);
  //   //within_above = _mm256_srai_epi16(within_above, 1);
  //   __m256i between = within_above;
  //   __m256i within = interpolate_red_row(&load_arr[j+width], width);
  //   for (int i = width; i < width*height-width; i += width) {
  //     if ((i/width)%2 == 1) {
  //       __m256i within_below = interpolate_blue_row(&load_arr[i+j+width], width);
  //       if (j+16 >= width-1) {
  //         _mm256_maskstore_epi32((int*)(&stor_arr[i+j+1]), remainder, within);
  //         _mm256_maskstore_epi32((int*)(&vert_arr[i+j-width+1]), remainder, between);
  //       } else {
  //         _mm256_maskstore_epi32((int*)(&stor_arr[i+j+1]), full_mask, within);
  //         _mm256_maskstore_epi32((int*)(&vert_arr[i+j-width+1]), full_mask, between);
  //       }

  //       between = _mm256_load_si256((__m256i*)(&load_arr[i+j+width+1]));
  //       __m256i above = _mm256_load_si256((__m256i*)(&load_arr[i+j-width+1]));
  //       between = _mm256_add_epi16(between, above);
  //       within_above = _mm256_add_epi16(within_above, within_below);

  //       between = _mm256_blend_epi16(between, within_above, 0b1010101010101010);

  //       within_above = within;
  //       within = within_below;
  //     } else {
  //       __m256i within_below = interpolate_red_row(&load_arr[i+j+width], width);
  //       if (j+16 >= width-1) {
  //         _mm256_maskstore_epi32((int*)(&stor_arr[i+j+1]), remainder, within);
  //         //_mm256_maskstore_epi32((int*)(&stor_arr[i+j-width+1]), remainder, within_above);
  //         _mm256_maskstore_epi32((int*)(&vert_arr[i+j-width+1]), remainder, between);
  //       } else {
  //         _mm256_maskstore_epi32((int*)(&stor_arr[i+j+1]), full_mask, within);
  //         //_mm256_maskstore_epi32((int*)(&stor_arr[i+j-width+1]), full_mask, within_above);
  //         _mm256_maskstore_epi32((int*)(&vert_arr[i+j-width+1]), full_mask, between);
  //       }

  //       between = _mm256_load_si256((__m256i*)(&load_arr[i+j+width+1]));
  //       __m256i above = _mm256_load_si256((__m256i*)(&load_arr[i+j-width+1]));
  //       between = _mm256_add_epi16(between, above);

  //       within_above = _mm256_add_epi16(within_above, within_below);

  //       between = _mm256_blend_epi16(within_above, between, 0b1010101010101010);

  //       within_above = within;
  //       within = within_below;
  //     }
      
  //   }

  //   if (j+16 >= width-1) {
  //       //_mm256_maskstore_epi32((int*)(&stor_arr[i+j-width+1]), remainder, within_above);
  //       _mm256_maskstore_epi32((int*)(&vert_arr[width*(height-2)+j+1]), remainder, between);
  //     } else {
  //       //_mm256_maskstore_epi32((int*)(&stor_arr[i+j-width+1]), full_mask, within_above);
  //       _mm256_maskstore_epi32((int*)(&vert_arr[width*(height-2)+j+1]), full_mask, between);
  //   }
  // }

  // for (int i = 2*width+width; i < width*height-width; i += 2*width) {
  //   for (int j = 0; j < width-1; j += 16) {
  //     __m256i avg_bg_up = interpolate_blue_row(&load_arr[i+j-width], width);
  //     __m256i avg_rg = interpolate_red_row(&load_arr[i+j], width);
  //     __m256i avg_bg_down = interpolate_blue_row(&load_arr[i+j+width], width);
  //     __m256i avg_bg = _mm256_add_epi16(avg_bg_up, avg_bg_down);
  //     avg_bg = _mm256_srai_epi16(avg_bg, 1);

  //     __m256i b_avg = _mm256_load_si256((__m256i*)&load_arr[i+j+width+1]);
  //     __m256i b_up = _mm256_load_si256((__m256i*)&load_arr[i+j-width+1]);
  //     b_avg = _mm256_avg_epu16(b_avg, b_up);

  //     b_avg = _mm256_blend_epi16(b_avg, avg_bg, -21846);

  //     if (j+16 >= width-1) {
  //       _mm256_maskstore_epi32((int*)(&stor_arr[i+j+1]), remainder, avg_rg);
  //       _mm256_maskstore_epi32((int*)(&stor_arr[i+j-width+1]), remainder, avg_bg_up);
  //       _mm256_maskstore_epi32((int*)(&stor_arr[i+j+width+1]), remainder, avg_bg_down);
  //       _mm256_maskstore_epi32((int*)(&vert_arr[i+j+1]), remainder, b_avg);
  //     } else {
  //       _mm256_maskstore_epi32((int*)(&stor_arr[i+j+1]), full_mask, avg_rg);
  //       _mm256_maskstore_epi32((int*)(&stor_arr[i+j-width+1]), full_mask, avg_bg_up);
  //       _mm256_maskstore_epi32((int*)(&stor_arr[i+j+width+1]), full_mask, avg_bg_down);
  //       _mm256_maskstore_epi32((int*)(&vert_arr[i+j+1]), full_mask, b_avg);
  //     }
  //     //printf("(%d, %d): ", i, j);
  //     //print_vec(total);
  //   }
  // }

  // for (int i = 2*width; i < width*height-width; i += 2*width) {
  //   for (int j = 0; j < width-1; j += 16) {
  //     __m256i avg_bg = interpolate_blue_row(&load_arr[i+j], width);

  //     if (j+16 >= width-1) {
  //       _mm256_maskstore_epi32((int*)(&stor_arr[i+j+1]), remainder, avg_bg);
  //     } else {
  //       _mm256_maskstore_epi32((int*)(&stor_arr[i+j+1]), full_mask, avg_bg);
  //     }
  //   }
  // }

  bilinear_avx(load_arr, stor_arr, vert_arr, width, height);

  printf("Interpolated Arrays:\n");
  for (int i = 0; i < height; i++) {
  	for (int j = 0; j < width; j++) {
  		printf("%d ", stor_arr[i*width+j]);
  	}
  	printf("\n");
  }
  printf("\n");
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      printf("%d ", vert_arr[i*width+j]);
    }
    printf("\n");
  }
  
  return 0;
}