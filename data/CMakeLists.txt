add_library(qtdemodata)
target_sources(qtdemodata
     PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/demoData.cpp
    ${CMAKE_CURRENT_LIST_DIR}/sampleData.qrc
)

set_target_properties(qtdemodata PROPERTIES
  AUTOMOC ON
  AUTOUIC ON
  AUTORCC ON
)

target_link_libraries(qtdemodata QtVis::qtvis compile_info)
target_include_directories(compile_info
  INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/data>
)
