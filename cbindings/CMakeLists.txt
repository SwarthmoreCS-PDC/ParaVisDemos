# single threaded CPU demo with C bindings
add_executable(simple simple.c)
target_link_libraries(simple QtVis::qtvis compile_info)
set_target_properties(simple PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
)

# pthreads demo with C bindings
if(CMAKE_USE_PTHREADS_INIT)
add_executable(pthr_simple pthread_c_demo.c)
target_link_libraries(pthr_simple QtVis::qtvis compile_info)
set_target_properties(pthr_simple PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
)

add_executable(edupar19 epar19.c)
target_link_libraries(edupar19 QtVis::qtvis compile_info)
set_target_properties(edupar19 PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
)
endif()
