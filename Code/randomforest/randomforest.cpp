// Example : random forest (tree) learning
// Author : Toby Breckon, toby.breckon@cranfield.ac.uk

// Copyright (c) 2011 School of Engineering, Cranfield University
// License : LGPL - http://www.gnu.org/licenses/lgpl.html

#include <cv.h>       // opencv general include file
#include <ml.h>		  // opencv machine learning include file
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>

#include <iostream>
#include <fstream>
#include <string>

#include <vector>


#define BOOST_FILESYSTEM_VERSION 3
#define BOOST_FILESYSTEM_NO_DEPRECATED
#include <boost/filesystem.hpp>

using namespace cv; // OpenCV API is in the C++ "cv" namespace
namespace fs = ::boost::filesystem;

/******************************************************************************/
// global definitions (for speed and ease of use)
//手写体数字识别

#define NUMBER_OF_TRAINING_SAMPLES 10000
#define ATTRIBUTES_PER_SAMPLE 784
#define NUMBER_OF_TESTING_SAMPLES 10000

#define NUMBER_OF_CLASSES 10

// N.B. classes are integer handwritten digits in range 0-9

/******************************************************************************/

// loads the sample database from file (which is a CSV text file)






// return the filenames of all files that have the specified extension
// in the specified directory and all subdirectories
void get_image_paths(const fs::path& root, const string& ext, std::vector<fs::path>& ret)
{
    if(!fs::exists(root) || !fs::is_directory(root)) return;

    fs::recursive_directory_iterator it(root);
    fs::recursive_directory_iterator endit;

    while(it != endit)
    {
        if(fs::is_regular_file(*it) && it->path().extension() == ext) ret.push_back(it->path().filename());
        ++it;

    }

}



int read_data_from_csv(const char* filename, Mat& data, Mat& classes,
                       int n_samples )
{
    float tmp;

    // if we can't read the input file then return 0
    FILE* f = fopen( filename, "r" );
    if( !f )
    {
        printf("ERROR: cannot read file %s\n",  filename);
        return 0; // all not OK
    }

    // for each sample in the file

    for(int line = 0; line < n_samples; line++)
    {
        // for each attribute on the line in the file
        for(int attribute = 0; attribute < (ATTRIBUTES_PER_SAMPLE + 1); attribute++)
        {
            if (attribute == 0)
            {
                // attribute is the class label {0 ... 9}
                fscanf(f, "%f,", &tmp);
                classes.at<float>(line, 0) = tmp;
                // printf("%f\n", classes.at<float>(line, 0));
            }
	    else if (attribute < ATTRIBUTES_PER_SAMPLE + 1)
            {
                // first 64 elements (0-63) in each line are the attributes
                fscanf(f, "%f,", &tmp);
                data.at<float>(line, attribute-1) = tmp;
                // printf("%f,", data.at<float>(line, attribute));
            }
        }
    }

    fclose(f);
    return 1; // all OK
}

/* the following code was copied from http://eric-yuan.me/cpp-read-mnist/ */
int reverse_int (int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return((int) ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

int read_mnist_images(const char* filename, Mat& data, int n_samples)
{
    std::ifstream file (filename, std::ios::binary);
    
    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        
        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = reverse_int(magic_number);
        
        file.read((char*) &number_of_images,sizeof(number_of_images));
        number_of_images = reverse_int(number_of_images);
        
        file.read((char*) &n_rows, sizeof(n_rows));
        n_rows = reverse_int(n_rows);
        
        file.read((char*) &n_cols, sizeof(n_cols));
        n_cols = reverse_int(n_cols);
        
        int size = n_rows * n_cols;
        
        if(number_of_images > n_samples)
        	number_of_images = n_samples;
        
        for(int i = 0; i < number_of_images; ++i)
        {
        	for(int pixel = 0; pixel < size; ++pixel)
            {
            	unsigned char temp = 0;
                file.read((char*) &temp, sizeof(temp));
            	data.at<float>(i, pixel) = temp;
            }
        }
    }
    
    return 1;
}

int read_mnist_labels(const char* filename, Mat& classes, int n_samples)
{
    std::ifstream file (filename, std::ios::binary);
    
    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        
        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = reverse_int(magic_number);
        
        file.read((char*) &number_of_images,sizeof(number_of_images));
        number_of_images = reverse_int(number_of_images);
        
        if(n_samples < number_of_images)
        	number_of_images = n_samples;
        
        for(int i = 0; i < number_of_images; ++i)
        {
            unsigned char temp = 0;
            file.read((char*) &temp, sizeof(temp));
            classes.at<float>(i, 0) = temp;
        }
    }
    
    return 1;
}


int read_image_files(const char* directory, Mat& data, int n_samples) {
	std::vector<fs::path> image_paths;
	get_image_paths(directory, "png", image_paths);

	int n_im = 0;
	for (std::vector<fs::path>::iterator it = image_paths.begin() ; it != image_paths.end(); ++it) {
		Mat image;
		image = imread(it->string(), 0);
		
		CV_Assert(image.channels() == 1);
		
		MatIterator_<uchar> mit;
		
		int n_pixel = 0;
		for( mit = image.begin<uchar>(); mit != image.end<uchar>(); ++mit) {
			data.at<float>(n_im, n_pixel++) = (float) *mit;
		}
		
		n_im++;
	}
	
	return 1;
}


