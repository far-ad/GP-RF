#include <opencv/cv.h>

#include <fstream>
#include <iostream>

template <typename T>
T *read_rgbd_data( const char* filename, int *rows, int *cols, int n_samples=0 ) {
  std::ifstream file( filename );

  if (!file.is_open())
  {
    std::cerr << "ERROR: could not open file \"" << filename << "\"!" << std::endl;
    return 0;
  }

  int n_size, n_attributes;
  file >> n_size >> n_attributes;

  n_samples = (n_samples > n_size || n_samples == 0) ? n_size : n_samples;

  // debugging output:
  std::cout << "samples: " << n_samples << std::endl;
  std::cout << "attributes: " << n_attributes << std::endl << std::endl;

  T *dataset = new T[n_samples*n_attributes];
  T tmp;

  // for each sample in the file
  for(int line = 0; line < n_samples; line++)
    {
      // for each attribute on the line in the file
      for(int attribute = 0; attribute < n_attributes; attribute++)
        {
	  file >> dataset[line*n_attributes + attribute];
        }
    }

  // return the dimension of the dataset
  *rows = n_samples;
  *cols = n_attributes;

  return dataset; // all OK
}

cv::Mat read_rgbd_data_cv( const char* filename, int n_samples=0 )
{
	int rows;
	int cols;
	float *dataset = read_rgbd_data<float>( filename, &rows, &cols, n_samples );

	return cv::Mat( rows, cols, CV_32FC1, dataset );
}

