#pragma once

#include <animator.h>
#include <cuda.h>

/* A CUDA Animation example that uses a separate
   user buffer on the GPU */
class UserBufferKernel: public Animator {

public:
  /* constructor sets dimensions of user grid */
  UserBufferKernel(int w, int h);
  ~UserBufferKernel();

  void update(ImageBuffer* img);

private:
  int* m_dev_grid; /* a grid in row major order */
  int m_rows, m_cols; /* dimensions of user grid */
};
