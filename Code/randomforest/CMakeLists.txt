find_package(OpenCV REQUIRED)

add_library(rfc rfc.cpp get_leaf_nodes.cpp)
target_link_libraries(rfc PUBLIC ${OpenCV_LIBS})

