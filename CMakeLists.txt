cmake_minimum_required(VERSION 3.11)

#CXX used for most of project, C for OSX pthread extensions
project(qtdemos LANGUAGES CXX C)

find_package(QtVis REQUIRED)

add_library(compile_info INTERFACE)
target_include_directories(compile_info
  INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)
target_compile_features(compile_info
  INTERFACE
    $<BUILD_INTERFACE:cxx_nullptr>
)
set(cxx_lang "$<COMPILE_LANGUAGE:CXX>")
set(debug_cxx_lang "$<AND:$<CONFIG:DEBUG>,${cxx_lang}>")
target_compile_options(compile_info INTERFACE
  $<${cxx_lang}:$<BUILD_INTERFACE:-Wall>>
  $<${debug_cxx_lang}:$<BUILD_INTERFACE:-Wunused-parameter>>
)

#Add user generated files here and name of executable
add_executable(cpuDemo main.cpp gradientVis.cpp)

#Link executable to vis library
target_link_libraries(cpuDemo QtVis::qtvis compile_info)

#Drop demo into top level build folder
set_target_properties(cpuDemo PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
)

if(CMAKE_USE_PTHREADS_INIT)
  add_subdirectory(pthreads)
endif()

if(OpenMP_FOUND OR OPENMP_FOUND)
  add_subdirectory(openMP)
endif()

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  #Autodetect CUDA architecture
  include(${CMAKE_SOURCE_DIR}/cmake_modules/detectCUDAarch.cmake)
  set(cuda_lang "$<COMPILE_LANGUAGE:CUDA>")
  set(debug_cuda_lang "$<AND:$<CONFIG:DEBUG>,${cuda_lang}>")
  target_compile_options(compile_info INTERFACE
    $<${cuda_lang}:$<BUILD_INTERFACE:-Xcompiler=-Wall>>
    $<${debug_cuda_lang}:$<BUILD_INTERFACE:-Xcompiler=-Wunused-parameter>>
  )
  target_compile_options(compile_info INTERFACE
      $<${cuda_lang}:$<BUILD_INTERFACE:${QVis_CUDA_ARCH}>>
  )
  #Sample cuda demos
  add_subdirectory(cuda)
else()
  message(STATUS
  "Could not find CUDA, set CMAKE_CUDA_COMPILER to\n
   full path to nvcc if you think this is an error")
  #Optionally, add path to nvcc to PATH env variable
endif()
