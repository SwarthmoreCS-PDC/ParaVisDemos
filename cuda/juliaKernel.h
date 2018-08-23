#pragma once

#include <animator.h>
#include <cuda.h>

/* A CUDA kernel example with a constructor and extra parameters
   Animates a Julia set fractal */
class JuliaKernel: public Animator {

public:
  /* set the initial real and imaginary seed values for animation */
  JuliaKernel(float re, float im):
    m_re(re), m_im(im), m_ticks(0){ };
  ~JuliaKernel() { /* do nothing */ };

  void update(ImageBuffer* img);

private:
  float m_re, m_im; /* Initial seed values for interation */
  int m_ticks;  /* animation time ticks */

};
