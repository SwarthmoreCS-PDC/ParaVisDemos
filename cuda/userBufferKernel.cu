#include "userBufferKernel.h"
#include "handle_cuda_error.h"
#include <unistd.h>

__global__ void int_to_color(color3 *optr, const int *my_cuda_data, int ncols);
__global__ void simplekernel(int *data, int ncols);

UserBufferKernel::UserBufferKernel(int w, int h):
  m_dev_grid(nullptr),
  m_rows(h), m_cols(w) {

  /* allocate some memory on the CPU first */
  int* cpu_data = new int[m_rows*m_cols];
  /* populate CPU memory with data  */
  for (int r = 0; r < h; r++) {
    for (int c = 0; c < w; c++) {
      cpu_data[r * w + c] = c;
    }
  }

  /* allocated GPU buffer */
  int bufSize = sizeof(int)*w*h;
  HANDLE_ERROR(
    cudaMalloc((void**)&m_dev_grid, bufSize));

  /* copy CPU data to GPU */
  HANDLE_ERROR(
    cudaMemcpy(m_dev_grid, cpu_data, bufSize, cudaMemcpyHostToDevice));

  /* after copy, data not needed on CPU */
  delete [] cpu_data; cpu_data = nullptr;
};

UserBufferKernel::~UserBufferKernel(){
  HANDLE_ERROR(cudaFree(m_dev_grid));
  m_dev_grid = nullptr;
}

void UserBufferKernel::update(ImageBuffer* img){

  dim3 blocks(m_cols / 8, m_rows / 8, 1);
  dim3 threads_block(8, 8, 1);

  // comment out the for loop to do a display update every
  // execution of simplekernel
  for (int i = 0; i < 90; i++){
    simplekernel<<<blocks, threads_block>>>(m_dev_grid, m_cols);
  }

  int_to_color<<<blocks, threads_block>>>(img->buffer, m_dev_grid,
                                          m_cols);

  // I needed to slow it down:
  usleep(90000);
}

// a kernel to set the color the opengGL display object based
// on the cuda data value
//
//  optr: is an array of openGL RGB pixels, each is a
//        3-tuple (x:red, y:green, z:blue)
//  my_cuda_data: is cuda 2D array of ints
__global__ void int_to_color(color3 *optr, const int *my_cuda_data, int cols) {

  // get this thread's block position to map into
  // location in opt and my_cuda_data
  // the x and y values depend on how you parallelize the
  // kernel (<<<blocks, threads>>>).
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int offset = x + y * cols;

  // change this pixel's color value based on some strange
  // functions of the my_cuda_data value
  optr[offset].r = (my_cuda_data[offset] + 10) % 255;  // R value
  optr[offset].g = (my_cuda_data[offset] + 100) % 255; // G value
  optr[offset].b = (my_cuda_data[offset] + 200) % 255; // B value
}

// a simple cuda kernel: cyclicly increases a points value by 10
//  data: a "2D" array of int values
__global__ void simplekernel(int *data, int ncols) {

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int offset = x + y * ncols;

  data[offset] = (data[offset] + 10) % 1000;
}
