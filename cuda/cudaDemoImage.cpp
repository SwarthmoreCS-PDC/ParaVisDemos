#include "imageFilter.h"
#include "demoData.h"
#include <qtViewer.h>
#include <dataVisCUDA.h>

/* A Demo of how to use the parallel grid data visualization toolkit
 */

int main(int argc, char *argv[]) {
  /* 1. Boilerplate code Create a QTViewer object with the argc, argv
     as the first arguments (needed by QT), and an optional title as the
     fifth argument */
  QTViewer viewer(argc, argv, 600, 500, "QtCUDA");

  /* 2. Dynamically create a derived instance of the
     DataVis class. In the case of CUDA examples, we
     derive from the DataVisCUDA class and specify the
     width and height of our desired grid, but a second step, 2b.
     is needed to connect a CUDA kernel to this class */
  /* load portable links to image data */
  loadQtVisDemoData();

  DataVisCUDA* vis = new DataVisCUDA(":/corgis.jpg");

  /* 2b. For CUDA based solutions, it is difficult to compile QtOpenGL
     and CUDA code in the same class, so we separate the process into
     a Qt DataVisCUDA component, which users should not need to change
     and a separate CUDA specific Animator class. Users can write new classes
     derived from the Animator class that use CUDA code as shown in the
     examples below */
  Animator* kern = new ImageFilter();

  /* 2b. CUDA users connect their kernel to DataVisCUDA class through the
     setAnimator method */
  vis->setAnimator(kern);

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
     causes problems. For CUDA users, however, the Animator is not automatically destroyed and must be cleaned up manually */
  /* TODO: See if this can be clarified, fixed, or if this is a good policy */
  delete kern; kern=nullptr;

  return res;
}
