add_library(cudademos
  juliaKernel.h rippleKernel.h userBufferKernel.h imageFilter.h
  juliaKernel.cu rippleKernel.cu userBufferKernel.cu imageFilter.cu
)
target_link_libraries(cudademos PRIVATE QtVis::qtviscuda compile_info)

add_executable(cudaDemo cudaDemo.cpp)

target_link_libraries(cudaDemo
      cudademos QtVis::qtvis QtVis::qtviscuda compile_info)
set_target_properties(cudaDemo PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
)

add_executable(cudaDemoImage cudaDemoImage.cpp)

target_link_libraries(cudaDemoImage
cudademos qtdemodata QtVis::qtvis QtVis::qtviscuda compile_info)
set_target_properties(cudaDemoImage PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
)
