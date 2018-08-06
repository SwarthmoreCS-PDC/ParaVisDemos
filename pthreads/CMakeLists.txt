add_executable(threadDemo threadDemo.cpp pthreadVis.h pthreadVis.cpp)
target_link_libraries(threadDemo QtVis::qtvis compile_info)

set_target_properties(threadDemo PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
)
