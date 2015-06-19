#include <gp-lvm/CMatrix.h>

// read the data from the RGBD-dataset into a CMatrix object
//cv::Mat read_rgbd_data( const char* filename, int n_samples=0 );
CMatrix *read_rgbd_data_cmat( const char* filename, int n_samples=0 );
