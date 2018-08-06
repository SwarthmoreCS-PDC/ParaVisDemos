if(NOT QVis_CUDA_ARCH)
  set(CUDA_TEST_SRC ${CMAKE_SOURCE_DIR}/test/detect_cuda_arch.cu)
  set(CUDA_TEST_OUT ${CMAKE_BINARY_DIR}/detect_cuda_arch)
  set(CUDA_CMD ${CMAKE_CUDA_COMPILER} -Wno-deprecated-gpu-targets -lcuda)
  if(CMAKE_CUDA_HOST_COMPILER)
    set(CUDA_CMD ${CUDA_CMD} -ccbin ${CMAKE_CUDA_HOST_COMPILER})
  endif()
  message(STATUS "${CUDA_CMD}")
  execute_process(COMMAND
    ${CUDA_CMD}
    ${CUDA_TEST_SRC} -o ${CUDA_TEST_OUT})
  execute_process(COMMAND ${CUDA_TEST_OUT}
                    RESULT_VARIABLE CUDA_RETURN_CODE
                    OUTPUT_VARIABLE ARCH)

    if(${CUDA_RETURN_CODE} EQUAL 0)
      set(CUDA_SUCCESS "TRUE")
    else()
      set(CUDA_SUCCESS "FALSE")
    endif()

    if (${CUDA_SUCCESS})
      message(STATUS "CUDA Architecture: ${ARCH}")
      #Yay! Add this to the cuda compile options
      set(QVis_CUDA_ARCH "${ARCH}" CACHE STRING "CUDA architecture flags")
      mark_as_advanced(QVis_CUDA_ARCH)
    else()
      message(WARNING "Unable to infer CUDA architecture: ${ARCH}")
    endif()
endif()
