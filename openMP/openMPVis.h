#pragma once

#include "dataVisCPU.h"
#include <omp.h>

/* An OpenMP gradient visualization */
class OpenMPVis: public DataVisCPU {

public:
  // Depth d currently not used
  OpenMPVis(int w, int h, int d=1);
  virtual ~OpenMPVis();

  void update();

private:
  int m_ticks;

};
