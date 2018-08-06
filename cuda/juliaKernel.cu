#include <cuda.h>
#include <iostream>
#include "juliaKernel.h"

__device__ int julia(int x, int y, int w, int h, float re, float im);
__global__ void kernel(color3 *ptr, int w, int h, float re, float im);

void JuliaKernel::update(ImageBuffer* img) {

  int w, h;
  w = img->width;
  h = img->height;
  dim3 grid(w, h);

  float im = m_im;
  float re = m_re;

  /* tweak seed by time to produce animation */
  im += 0.2 * sin(m_ticks/20.);
  re += 0.3 * cos(m_ticks/17.);

  /* call kernel */
  kernel<<<grid, 1>>>(img->buffer, w, h, re, im);

  /* update time */
  m_ticks = (m_ticks+1)%1234;
}

struct cuComplex {
  float r;
  float i;
  __device__ cuComplex(float a, float b) : r(a), i(b) {}
  __device__ float magnitude2(void) { return r * r + i * i; }
  __device__ cuComplex operator*(const cuComplex &a) {
    return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
  }
  __device__ cuComplex operator+(const cuComplex &a) {
    return cuComplex(r + a.r, i + a.i);
  }
};

__device__ int julia(int x, int y, int w, int h, float re, float im) {
  const float scale = 1.5;
  float jx = scale * (float)(w / 2 - x) / (h / 2);
  float jy = scale * (float)(h / 2 - y) / (h / 2);

  cuComplex c(re, im);
  cuComplex a(jx, jy);

  int i = 0;
  for (i = 0; i < 200; i++) {
    a = a * a + c;
    if (a.magnitude2() > 1000) {
      return 0;
    }
  }

  return 1;
}

__global__ void kernel(color3 *ptr, int w, int h, float re, float im) {
  // map from threadIdx/BlockIdx to pixel position
  int x = blockIdx.x;
  int y = blockIdx.y;
  int offset = x + y * gridDim.x;

  // now calculate the value at that position
  int juliaValue = julia(x, y, w, h, re, im);
  ptr[offset].r = 255 * juliaValue;
  ptr[offset].g = 0;
  ptr[offset].b = 64 * (1 - juliaValue);
}
