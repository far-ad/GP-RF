#include <opencv/cv.h>
#include "../dataset/load_rgbd_cv.h"
#include "gpc.hpp"

#include <iostream>

void fvec2dvec(float* vec, double* res_vec, int n)
{
	for(int i=0; i<n; i++) {
		res_vec[i] = (double) vec[i];
	}
}


int main(int argc, char** argv) {

	cv::Mat training_data = read_rgbd_data_cv(argv[1]);
	cv::Mat training_labels = extract_label(read_rgbd_data_cv(argv[2]));
	cv::Mat testing_data = read_rgbd_data_cv(argv[3]);
	cv::Mat testing_labels = extract_label(read_rgbd_data_cv(argv[4]));

	int input_dim = training_data.cols;
	int n_training_data = training_data.rows;
	int n_testing_data = testing_data.rows;

	double* training_data_d = new double[n_training_data*input_dim];
	double* training_labels_d = new double[n_training_data*input_dim];
	double* testing_data_d = new double[n_testing_data*input_dim];
	double* testing_labels_d = new double[n_testing_data*input_dim];

	fvec2dvec((float*) training_data.datastart, training_data_d, n_training_data);
	fvec2dvec((float*) training_labels.datastart, training_labels_d, n_training_data);
	fvec2dvec((float*) testing_data.datastart, testing_data_d, n_testing_data);
	fvec2dvec((float*) testing_labels.datastart, testing_labels_d, n_testing_data);

	GPC classifier(input_dim);
	classifier.train(training_labels_d, training_data_d, n_training_data);
	
	for(int i=0; i<n_testing_data; i++)
	{
		std::cout << "class: " << testing_labels.at<float>(i,0) << std::endl;
		std::cout << "\tprediction: " << classifier.predict(&testing_data_d[input_dim*i]) << std::endl;
	}
}

