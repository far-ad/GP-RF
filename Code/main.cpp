#include <stdio.h>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <string>
#include <vector>
#include <fstream>
#include "randomforest/rfc.hpp"
//#include "gp/gpc.hpp"


using namespace std;


double* read_rgbd_data( const char* filename, int n_samples=0 )
{
    std::ifstream file( filename );

     if (!file.is_open())
     {
       std::cerr << "ERROR: could not open file!" << std::endl;
       return NULL;
     }

    int n_size, n_attributes;
    file >> n_size >> n_attributes;

    n_samples = (n_samples > n_size || n_samples == 0) ? n_size : n_samples;

    // debugging output:
    std::cout << "samples: " << n_samples << std::endl;
    std::cout << "attributes: " << n_attributes << std::endl << std::endl;

    // cv::Mat dataset = cv::Mat(n_samples, n_attributes, CV_32FC1);
    double tmp;
    vector<double> dataset;
    // for each sample in the file

    for(int line = 0; line < n_samples; line++)
    {
      // for each attribute on the line in the file
      for(int attribute = 0; attribute < n_attributes; attribute++)
      {
        file >> tmp;
	    dataset.push_back(tmp);
      }
    }
    return dataset.data(); // all OK
}


int main(int argc, char** argv){

    if(argc < 5) {
		std::cout << "please provide training data/labels and testing data/labels as arguments!" << std::endl;
		exit(1);
	}

	int max_samples = 2;

	// use fifth argument as maximum number of samples if the argument is present
	if(argc > 5) {

		istringstream(argv[5]) >> max_samples;
	}

	// read the input data into cv matrices to demonstrate the conversion to double*
	double* training_data = read_rgbd_data(argv[1], max_samples);
	double* training_labels = read_rgbd_data(argv[2], max_samples);
	double* testing_data = read_rgbd_data(argv[3], max_samples);
	double* testing_labels = read_rgbd_data(argv[4], max_samples);
	
    printf("one data from: %lf", training_data[10]);
	// test if class rfc works
	/*int n_features = 14000; 
	int n_samples = 200;
	std::list<leaf_samples> leaf_with_samples;
	RFC rfclassifier(n_features);
	rfclassifier.train(training_labels, training_data, n_samples);
	leaf_with_samples = rfclassifier.split_data_by_leafs(training_data, n_samples);
	printf("size of list of leaf_with_samples:%lu \n", leaf_with_samples.size());*/
	


}
