#include <qtViewer.h>
#include "gradientVis.h"

/* A Demo of how to use the parallel grid data visualization toolkit
 */

int main(int argc, char *argv[]) {
  /* 1. Boilerplate code Create a QTViewer object with the argc, argv
     as the first arguments (needed by QT), and an optional title as the
     fifth argument */
  /* TODO: use width, height to control size of window */
  QTViewer viewer(argc, argv, 10, 10, "QtCPU");

  /* 2. Dynamically create a derived instance of the
     DataVis class. In the case of CUDA examples, we
     derive from the DataVisCUDA class and specify the
     width and height of our desired grid, but a second step, 2b.
     is needed to connect a CUDA kernel to this class */
  int width = 50;
  int height = 50;

  DataVis* vis = new GradientVis(width,height);

  /* 3. All users must inform the viewer of their DataVis animation with
     the setAnimation method */
  viewer.setAnimation(vis);

  /* 4. Once everything is set up, calling run() will enter the
     Qt/OpenGL event loop and animate the scene using the visualization
     provided by the user */
  int res = viewer.run();

  /* Cleanup */
  /* The viewer only returns once the OpenGL context has been destroyed
     currently, the viewer will delete the vis object on the user behalf,
     since attempting to delete it after the OpenGL context has been destroyed
     causes problems.*/
  return res;
}
