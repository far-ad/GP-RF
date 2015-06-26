#include "gpc.hpp"
#include "GP_RF.hpp"
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <string>
#include "../dataset/load_rgbd.h"

int main(int argc, char** argv) {

	if(argc < 5) {
		std::cout << "please provide training data/labels and testing data/labels as arguments!" << std::endl;
		exit(1);
	}

	int max_samples = 0;

	// use fifth argument as maximum number of samples if the argument is present
	if(argc > 5) {
		istringstream(argv[5]) >> max_samples;
	}

	int n_training_samples, n_testing_samples, n_features;
	int dummy;

	double *training_data = read_rgbd_data<double>( argv[1], &n_training_samples, &n_features, max_samples );
	double *training_labels = read_rgbd_data<double>( argv[2], &n_training_samples, &dummy, max_samples );
	double *testing_data = read_rgbd_data<double>( argv[3], &n_testing_samples, &n_features, max_samples );
	double *testing_labels = read_rgbd_data<double>( argv[4], &n_testing_samples, &dummy, max_samples );


	GP_RF* classifier = new GP_RF(n_features);
	classifier->train(training_labels, training_data, n_training_samples);

}
