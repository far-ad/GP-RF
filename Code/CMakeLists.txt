cmake_minimum_required (VERSION 2.6)

project (rdf-gp)

add_subdirectory(randomforest)
add_subdirectory(gp)

add_library(rdfgp GP_RF.cpp)
target_link_libraries(rdfgp rfc gpc)

add_executable(rdf-gp GP_RF_test.cpp)
target_link_libraries(rdf-gp rdfgp)

