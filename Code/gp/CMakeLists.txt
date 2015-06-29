include_directories ("/usr/include/gp-lvm" "/usr/local/include/gp-lvm")

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

find_library(GFORTRAN_LIB NAMES gfortran)
find_library(GP_LIB NAMES gp)

add_library(gpc gpc.cpp)
target_link_libraries(gpc ${GP_LIB} ${GFORTRAN_LIB} ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})

