#include <opencv/cv.h>
#include "../dataset/load_rgbd_cv.h"
#include "gpc.hpp"

#include <iostream>
#include <sstream>
#include <cstdlib>
#include <string>

void fvec2dvec(float* vec, double* res_vec, int n)
{
	for(int i=0; i<n; i++) {
		res_vec[i] = (double) vec[i];
	}
}


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

	// read the input data into cv matrices to demonstrate the conversion to double*
	cv::Mat training_data = read_rgbd_data_cv(argv[1], max_samples);
	cv::Mat training_labels = read_rgbd_data_cv(argv[2], max_samples);
	cv::Mat testing_data = read_rgbd_data_cv(argv[3], max_samples);
	cv::Mat testing_labels = read_rgbd_data_cv(argv[4], max_samples);

	int input_dim = training_data.cols;
	int n_training_data = training_data.rows / 2;
	int n_testing_data = testing_data.rows;

	double* training_data_d = new double[n_training_data*input_dim];
	double* training_labels_d = new double[n_training_data*input_dim];
	double* testing_data_d = new double[n_testing_data*input_dim];
	double* testing_labels_d = new double[n_testing_data*input_dim];

	// for now we just copy the content to a different vector for conversion
	// The better approach would be to keep the opencv matrices as doubles, s.t.
	// we can take their data directly
	fvec2dvec((float*) training_data.datastart, training_data_d, n_training_data);
	fvec2dvec((float*) training_labels.datastart, training_labels_d, n_training_data);
	fvec2dvec((float*) testing_data.datastart, testing_data_d, n_testing_data);
	fvec2dvec((float*) testing_labels.datastart, testing_labels_d, n_testing_data);

	// atm the gaussian classifier can only classify in a binary way
	double target_label = 2;

	GPC classifier(input_dim, target_label);
	classifier.train(training_labels_d, training_data_d, n_training_data);

	classifier.test(testing_labels_d, testing_data_d, n_testing_data);

	std::string dummy;
	std::getline(std::cin, dummy);

	fvec2dvec(((float*) training_data.datastart) + n_training_data, training_labels_d, n_training_data);
	fvec2dvec(((float*) training_labels.datastart) + n_training_data, training_labels_d, n_training_data);
	
	classifier.train(training_labels_d, training_data_d, n_training_data);
	classifier.test(testing_labels_d, testing_data_d, n_testing_data);
}

