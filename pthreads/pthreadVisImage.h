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
class PThreadVisImage: public DataVisCPU {

public:
  // Load initial dimensions, colors from image
  PThreadVisImage(int numThreads, QString fileName);
  virtual ~PThreadVisImage();

  void update();

private:
  int m_numThreads;
  pthread_t* m_threads;
  threadInfo* m_tinfo;
  pthread_barrier_t m_barrier;

};
