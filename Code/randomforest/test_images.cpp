#include "randomforest.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cv.h>

#define ATTRIBUTES_PER_SAMPLE 2500

using namespace cv;

int main(int argc, char** argv) {
	char* directory = argv[1];
	Mat training_data = Mat(10, ATTRIBUTES_PER_SAMPLE, CV_32FC1);

	read_image_files(directory, training_data, 10);
	namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
	for(int i=0; i<training_data.rows; i++) {
	  Mat image = training_data.row(i).reshape(0,50);
	  imshow( "Display window", image );                   // Show our image inside it.

       	waitKey(0);
	}   
}
