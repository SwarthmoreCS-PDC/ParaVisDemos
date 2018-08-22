#pragma once

#include <dataVisCPU.h>
#include <pthread.h>

#ifdef __APPLE__
#include <osx/pthread_barrier.h>
#endif

typedef struct {
  int nThreads;
  int id;
  ImageBuffer* img;
  pthread_barrier_t* barrier;
} threadInfo;

/* A PThreads Demo */
class PThreadVis: public DataVisCPU {

public:
  // Depth d currently not used
  PThreadVis(int numThreads, int w, int h, int d=1);
  virtual ~PThreadVis();

  void update();

private:
  int m_numThreads;
  pthread_t* m_threads;
  threadInfo* m_tinfo;
  pthread_barrier_t m_barrier;

};
