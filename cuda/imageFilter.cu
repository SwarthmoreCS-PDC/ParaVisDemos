#include <cuda.h>
#include <iostream>
#include <unistd.h>
#include "imageFilter.h"

__global__ void kernel(color3 *ptr, int w, int h, int ticks);

void ImageFilter::update(ImageBuffer* img) {
  int tdim = 8; // number of threads in x/y direction per block

  int w = img->width;
  int h = img->height;

  /* set up grid dimension */
  dim3 blocks((w+(tdim-1)) / tdim, (h+(tdim-1)) / tdim);
  /* set up block dimension */
  dim3 threads_block(tdim, tdim);

  /* call the CUDA kernel with grid dimension */
  kernel<<<blocks, threads_block>>>(img->buffer, w, h, m_ticks);
  usleep(500000);
  printf("%d\n",m_ticks);
  /* step size controls speed of animation */
  m_ticks += 1;
}

__global__ void kernel(color3 *ptr, int w, int h, int ticks) {
  // map from threadIdx/BlockIdx to pixel position
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * w;

  // compute distance from center of image
  float fx = x - w / 2;
  float fy = y - h / 2;
  float d = sqrtf(fx * fx + fy * fy);

  // use distance to modulate grey value intensity
  int grey = (int) (20.0f * cos(d / 10.0f - ticks / 7.0f) / (d / 10.0f + 1.0f));

  if(x<w && y<h){
  if(ptr[offset].r<200){
    ptr[offset].r += 10;
  }
  if(ptr[offset].r>210){
    ptr[offset].r -= 60;
  }
  if(ptr[offset].b<200){
    ptr[offset].b += 5;
  }
  if(ptr[offset].b>210){
    ptr[offset].b += 70;
  }
  }
}
