add_library(cudademos
  juliaKernel.h rippleKernel.h userBufferKernel.h imageFilter.h
  juliaKernel.cu rippleKernel.cu userBufferKernel.cu imageFilter.cu
)
target_link_libraries(cudademos PRIVATE QtVis::qtviscuda compile_info)
set_target_properties(cudademos PROPERTIES CUDA_ARCHITECTURES "50;52;61;72")

add_executable(cudaDemo cudaDemo.cpp)
target_link_libraries(cudaDemo
      cudademos QtVis::qtvis QtVis::qtviscuda compile_info)


add_executable(cudaDemoImage cudaDemoImage.cpp)
target_link_libraries(cudaDemoImage
cudademos qtdemodata QtVis::qtvis QtVis::qtviscuda compile_info)


set_target_properties(cudaDemo cudaDemoImage
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
)
