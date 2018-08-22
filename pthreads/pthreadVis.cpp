#include "pthreadVis.h"
#include <iostream>

void *threadUpdate(void* info){
  threadInfo* tinfo = (threadInfo*) info;
  int off;
  int w,h;
  int rowstart, rowstop, maxrows;
  int ticks = 0;
  unsigned char val;
  w= tinfo->img->width;
  h= tinfo->img->height;

  maxrows = h/tinfo->nThreads;
  if(h%tinfo->nThreads >0) { maxrows++; }
  rowstart=maxrows*tinfo->id;
  rowstop=rowstart+maxrows;
  if(rowstop > h) { rowstop = h; }

  while(true){
    for(int r=rowstart; r<rowstop; r++){
      for(int c=0; c<w; c++){
        off = r*w+c;
        val = (unsigned char) (128. * r /maxrows);
        val = (val+ticks)%128;
        tinfo->img->buffer[off].r=val;
        tinfo->img->buffer[off].g=0;
        tinfo->img->buffer[off].b=128-val;
      }
    }
    pthread_barrier_wait(tinfo->barrier);
    ticks++;
  }
  return nullptr;
}

PThreadVis::PThreadVis(int numThreads, int w, int h, int d) :
   DataVisCPU(w,h,d), m_numThreads(numThreads),
   m_threads(nullptr), m_tinfo(nullptr) {
  int i;
  m_threads = new pthread_t[m_numThreads];
  m_tinfo = new threadInfo[m_numThreads];
  m_tinfo[0].nThreads=m_numThreads;
  m_tinfo[0].img=&m_image;
  m_tinfo[0].barrier=&m_barrier;
  pthread_barrier_init(&m_barrier, nullptr, m_numThreads+1);
  for(i=0;i<m_numThreads;i++){
    m_tinfo[i]=m_tinfo[0];
    m_tinfo[i].id=i;
    pthread_create(&m_threads[i], nullptr, threadUpdate, (void*)&m_tinfo[i]);
  }
};

PThreadVis::~PThreadVis(){
  pthread_barrier_wait(&m_barrier);
  pthread_barrier_destroy(&m_barrier);
  delete [] m_threads; m_threads=nullptr;
  delete [] m_tinfo; m_tinfo=nullptr;
}

void PThreadVis::update() {
  pthread_barrier_wait(&m_barrier);
}
