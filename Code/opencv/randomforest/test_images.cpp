#include "randomforest.h"
#include <cv.h>

#define ATTRIBUTES_PER_SAMPLE 2500

int main(int argc, char** argv) {
	char* directory = argv[1];
	Mat training_data = Mat(10, ATTRIBUTES_PER_SAMPLE, CV_32FC1);

	read_image_files(directory, training_data, 10);

	for(int i=0; i<training_data.rows; i++) {
	}

	namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
        imshow( "Display window", image );                   // Show our image inside it.

	waitKey(0);
}
