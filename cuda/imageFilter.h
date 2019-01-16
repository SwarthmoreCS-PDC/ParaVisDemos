#pragma once

#include <animator.h>
#include <cuda.h>

/* A CUDA Animation that displays a radial
   greyscale ripple effect  */
class ImageFilter: public Animator {

public:
  ImageFilter(): m_ticks(0){ };
  ~ImageFilter() { /* do nothing */ };

  void update(ImageBuffer* img);

private:
  int m_ticks; /* number of timesteps in animation */

};
