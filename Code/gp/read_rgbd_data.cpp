#include <opencv/cv.h>

#include <fstream>
#include <iostream>

//for rf
cv::Mat read_rgbd_data( const char* filename, int n_samples=0 )
{
  std::ifstream file( filename );

  if (!file.is_open())
  {
    std::cerr << "ERROR: could not open file!" << std::endl;
    return cv::Mat();
  }

  int n_size, n_attributes;
  file >> n_size >> n_attributes;

  n_samples = (n_samples > n_size || n_samples == 0) ? n_size : n_samples;

  // debugging output:
  std::cout << "samples: " << n_samples << std::endl;
  std::cout << "attributes: " << n_attributes << std::endl << std::endl;

  cv::Mat dataset = cv::Mat(n_samples, n_attributes, CV_32FC1);
  float tmp;

  // for each sample in the file

  for(int line = 0; line < n_samples; line++)
    {
      // for each attribute on the line in the file
      for(int attribute = 0; attribute < n_attributes; attribute++)
        {
	  file >> tmp;
	  dataset.at<float>(line, attribute) = tmp;
        }
    }

  return dataset; // all OK
}





