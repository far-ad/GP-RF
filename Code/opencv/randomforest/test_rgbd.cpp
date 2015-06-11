#include <opencv/cv.h>

cv::Mat read_rgbd_data( const char* filename, int n_samples );

int main(int argc, char** argv)
{
  cv::Mat dataset, labels;
  dataset = read_rgbd_data(argv[1],10);
  labels = read_rgbd_data(argv[2], 10);
}  
