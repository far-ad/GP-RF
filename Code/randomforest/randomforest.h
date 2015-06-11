#include <opencv/cv.h>

using namespace cv;

int read_data_from_csv(const char* filename, Mat& data, Mat& classes,
                       int n_samples );
int read_mnist_images(const char* filename, Mat& data, int n_samples);
int read_mnist_labels(const char* filename, Mat& classes, int n_samples);
int read_image_files(const char* directory, Mat& data, int n_samples);
