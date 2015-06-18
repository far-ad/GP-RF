#include <opencv/cv.h>
#include "load_rgbd.h"

cv::Mat read_rgbd_data_cv( const char* filename, int n_samples=0 )
{
	int rows;
	int cols;
	float *dataset = read_rgbd_data<float>( filename, &rows, &cols, n_samples );

	return cv::Mat( rows, cols, CV_32FC1, dataset );
}

