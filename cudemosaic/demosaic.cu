#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include "cu_errchk.h"

// red = even row, even col
// blue = odd row, odd col
// green1 = even row, odd col green
// green2 = odd row, evel col green
typedef enum { RED = 0, GREEN1 = 1, GREEN2 = 2, BLUE = 3 } color_t;

__device__ __inline__ color_t fc(int row, int col) {
    // even row + odd col = blue, odd row + even col = red. match = green.
    return (color_t) (((row & 1) << 1) + (col & 1));
}
/* R G R G 
 * G B G B 
 * R G R G 
 * G B G B 
 */
__global__ void linear_demosaic_r(ushort *raw_in, int width, int height,
                                  ushort4 *out) {
    int row = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
    int col = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    
    if (row < 1 || col < 1 || row >= (height - 1) || col >= (width - 1))
        return;

    // shared buffers: uncomment and use this when the whole thing works without it.
    /*
    int local_row = threadIdx.y * 2 + 1;
    int local_col = threadIdx.x * 2;
    int local_width = 2 * blockDim.x;
    int local_height = 2 * blockDim.y;*/
    /* fill a shared buffer of raw values.
     * R G
     * G B
     * we have a thread for each red, so each thread fills in its own element,
     * and the ones above it and to its right.
     */
    /*
    __shared__ ushort in_buf[4 * blockDim.x * blockDim.y];
    in_buf[(local_row - 1) * local_width + local_col] =
        raw_in[(row - 1) * width + col];
    in_buf[(local_row - 1) * local_width + (local_col + 1)] =
        raw_in[(row - 1) * width + (col + 1)];
    in_buf[local_row * local_width + local_col] = raw_in[row * width + col];
    in_buf[local_row * local_width + (local_col + 1)] =
        raw_in[row * width + (col + 1)];
    */
    
    ushort4 px;
    px.x = raw_in[row * width + col]; //red: this pixel
    // green: average of pixels directly adjacent to us
    px.y = (raw_in[(row - 1) * width + col] +
            raw_in[(row + 1) * width + col] +
            raw_in[row * width + (col - 1)] +
            raw_in[row * width + (col + 1)])
           / 4;
    // blue: average of pixels diagonally adjacent to us
    px.z = (raw_in[(row - 1) * width + (col - 1)] +
            raw_in[(row - 1) * width + (col + 1)] +
            raw_in[(row + 1) * width + (col - 1)] +
            raw_in[(row + 1) * width + (col + 1)])
           / 4;
    //printf("(%d, %d) r:  (%d, %d, %d)\n", row, col, px.x, px.y, px.z);
    out[row * width + col] = px;
}


__global__ void linear_demosaic_g1(ushort *raw_in, int width, int height,
                                  ushort4 *out) {
    int row = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
    int col = (blockIdx.x * blockDim.x + threadIdx.x) * 2 + 1;

    if (row < 1 || col < 1 || row >= (height - 1) || col >= (width - 1))
        return;
    
    /*
    int local_row = threadIdx.y;
    int local_col = threadIdx.x * 2 + (row & 1);
    int local_width = 2 * blockDim.x;
    int local_height = blockDim.y;
    __shared__ ushort in_buf[2 * blockDim.x * blockDim.y];
    */  

    ushort4 px;
    // red: left and right average
    px.x = (raw_in[row * width + (col - 1)] +
            raw_in[row * width + (col + 1)])
        / 2;
    // green: this pixel
    px.y = raw_in[row * width + col];
    // blue: above and below average
    px.z = (raw_in[(row - 1) * width + col] +
            raw_in[(row + 1) * width + col])
        / 2;
    //printf("(%d, %d) g1: (%d, %d, %d)\n", row, col, px.x, px.y, px.z);
    out[row * width + col] = px;
}

__global__ void linear_demosaic_g2(ushort *raw_in, int width, int height,
                                  ushort4 *out) {
    int row = (blockIdx.y * blockDim.y + threadIdx.y) * 2 + 1;
    int col = (blockIdx.x * blockDim.x + threadIdx.x) * 2;

    if (row < 1 || col < 1 || row >= (height - 1) || col >= (width - 1))
        return;
    
    /*
    int local_row = threadIdx.y;
    int local_col = threadIdx.x * 2 + (row & 1);
    int local_width = 2 * blockDim.x;
    int local_height = blockDim.y;
    __shared__ ushort in_buf[2 * blockDim.x * blockDim.y];
    */  

    ushort4 px;

    // red: above and below average
    px.x = (raw_in[(row - 1) * width + col] +
            raw_in[(row + 1) * width + col])
        / 2;
    // green: this pixel
    px.y = raw_in[row * width + col];
    // blue: left and right average
    px.z = (raw_in[row * width + (col - 1)] +
            raw_in[row * width + (col + 1)])
        / 2;
    //printf("(%d, %d) g2: (%d, %d, %d)\n", row, col, px.x, px.y, px.z);
    out[row * width + col] = px;
}

__global__ void linear_demosaic_b(ushort *raw_in, int width, int height,
                                  ushort4 *out) {
    int row = (blockIdx.y * blockDim.y + threadIdx.y) * 2 + 1;
    int col = (blockIdx.x * blockDim.x + threadIdx.x) * 2 + 1;

    if (row < 1 || col < 1 || row >= (height - 1) || col >= (width - 1))
        return;

    ushort4 px;

    // red: average of pixels diagonally adjacent to us
    px.x = (raw_in[(row - 1) * width + (col - 1)] +
            raw_in[(row - 1) * width + (col + 1)] +
            raw_in[(row + 1) * width + (col - 1)] +
            raw_in[(row + 1) * width + (col + 1)])
          / 4;
    // green: average of pixels directly adjacent to us
    px.y = (raw_in[(row - 1) * width + col] +
            raw_in[(row + 1) * width + col] +
            raw_in[row * width + (col - 1)] +
            raw_in[row * width + (col + 1)])
          / 4;
    px.z = raw_in[row * width + col]; //blue: this pixel
    //printf("(%d, %d) b:  (%d, %d, %d)\n", row, col, px.x, px.y, px.z);
    out[row * width + col] = px;
}

inline int updiv(int n, int d) {
    return (n + d - 1) / d;
}

cudaStream_t colorStreams[4];
void initStreams() {
    for (int i = 0; i < 4; i++) {
        cudaStreamCreate(&colorStreams[i]);
    }
}

void linear_demosaic_cu(ushort *raw_in, ushort *img_out, int width, int height) {
    ushort *deviceIn;
    ushort4 *deviceOut;

    // initialize buffers
    cudaMalloc(&deviceIn, sizeof(ushort) * width * height);
    cudaMalloc(&deviceOut, sizeof(ushort4) * width * height);
    cudaMemcpy(deviceIn, raw_in, sizeof(ushort) * width * height,
	       cudaMemcpyHostToDevice);

    // launch kernels
    const int blkSide = 16;
    dim3 blkDim(blkSide, blkSide);
    dim3 gridDim(updiv(width, blkSide * 2), updiv(height, blkSide * 2));

    //printf("launch? %d %d %d %d\n", blkDim.x, blkDim.y, gridDim.x, gridDim.y);
    linear_demosaic_r<<<gridDim, blkDim, 0, colorStreams[RED]>>>(deviceIn, width, height, deviceOut);
    //gpuErrchk( cudaPeekAtLastError() );
    linear_demosaic_g1<<<gridDim, blkDim, 0, colorStreams[GREEN1]>>>(deviceIn, width, height, deviceOut);
    //gpuErrchk( cudaPeekAtLastError() );
    linear_demosaic_g2<<<gridDim, blkDim, 0, colorStreams[GREEN2]>>>(deviceIn, width, height, deviceOut);
    //gpuErrchk( cudaPeekAtLastError() );
    linear_demosaic_b<<<gridDim, blkDim, 0, colorStreams[BLUE]>>>(deviceIn, width, height, deviceOut);
    //gpuErrchk( cudaPeekAtLastError() );
    //gpuErrchk( cudaDeviceSynchronize() );
    cudaDeviceSynchronize();

    // copy results & free buffers
    cudaMemcpy(img_out, deviceOut, sizeof(ushort4) * width * height,
               cudaMemcpyDeviceToHost);
    cudaFree(deviceIn);
    cudaFree(deviceOut);
}