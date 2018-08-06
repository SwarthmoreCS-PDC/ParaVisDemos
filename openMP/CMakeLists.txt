add_executable(openMPDemo openMPDemo.cpp)
target_sources(openMPDemo
  PRIVATE
    openMPVis.h openMPVis.cpp
)
target_link_libraries(openMPDemo QtVis::qtvis compile_info)

set_target_properties(openMPDemo PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
)
