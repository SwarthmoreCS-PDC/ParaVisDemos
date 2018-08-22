#include "openMPVis.h"

OpenMPVis::OpenMPVis(int w, int h, int d) :
   DataVisCPU(w,h,d), m_ticks(0){
   /* do nothing */
};

OpenMPVis::~OpenMPVis(){
  /* do nothing */
}

void OpenMPVis::update() {
  int off;
  unsigned char val;
  int c;
  #pragma omp parallel for private(c, off, val)
  for(int r=0; r<m_height; r++){
    for(c=0; c<m_width; c++){
      off = r*m_width+c;
      val = (unsigned char) (128. * r / m_height);
      val = (val+m_ticks)%128;
      m_image.buffer[off].r=val;
      m_image.buffer[off].g=0;
      m_image.buffer[off].b=128-val;
    }
  }
  m_ticks += 1;
}
