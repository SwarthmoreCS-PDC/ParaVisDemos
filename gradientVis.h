#pragma once

#include "dataVisCPU.h"

/* A single threaded gradient visualization */
class GradientVis: public DataVisCPU {

public:
  // Depth d currently not used
  GradientVis(int w, int h, int d=1);
  virtual ~GradientVis();

  void update();

private:
  int m_ticks;

};
