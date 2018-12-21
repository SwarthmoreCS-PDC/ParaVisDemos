#include "pthreadVisImage.h"
#include <iostream>

void copyRow(color3* src, color3* dst, int w){
  for(int c=0; c<w; c++){
    dst[c] = src[c];
  }
}

void *threadUpdate(void* info){
  threadInfo* tinfo = (threadInfo*) info;
  int off;
  int w,h;
  int rowstart, rowstop, maxrows;
  int ticks = 0;
  w= tinfo->img->width;
  h= tinfo->img->height;

  maxrows = h/tinfo->nThreads;
  if(h%tinfo->nThreads >0) { maxrows++; }
  rowstart=maxrows*tinfo->id;
  rowstop=rowstart+maxrows;
  if(rowstop > h) { rowstop = h; }

  color3* oldrow = new color3[w];

  while(true){
    off = rowstart*w;
    copyRow(&tinfo->img->buffer[off], oldrow,w);
    for(int r=rowstart+1; r<rowstop; r++){
      off = r*w;
      copyRow(&tinfo->img->buffer[off], &tinfo->img->buffer[off-w],w);
    }
    off = (rowstop-1)*w;
    copyRow(oldrow, &tinfo->img->buffer[off],w  );
    pthread_barrier_wait(tinfo->barrier);
    ticks++;
  }

  delete [] oldrow;
  oldrow = nullptr;
  return nullptr;
}

PThreadVisImage::PThreadVisImage(int numThreads, QString fileName) :
   DataVisCPU(fileName), m_numThreads(numThreads),
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

PThreadVisImage::~PThreadVisImage(){
  pthread_barrier_wait(&m_barrier);
  pthread_barrier_destroy(&m_barrier);
  delete [] m_threads; m_threads=nullptr;
  delete [] m_tinfo; m_tinfo=nullptr;
}

void PThreadVisImage::update() {
  pthread_barrier_wait(&m_barrier);
}
