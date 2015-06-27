#include <map>
#include <opencv/ml.h>
#include "gpc.hpp"
#include "rfc.hpp"


using namespace std;

class GP_RF{
public:
	GP_RF(int n_features);


	void train(double *training_labels, double *training_data, int n_samples);

	///returns predicted label
	double predict(double *observation);

	///returns error rate
	double test(double *testing_labels, double *testing_data, int n_samples);

private:
	int n_features;
	RFC *forest_classifier;
	///mapping from the leaf nodes to the Gaussian Process classifiers
	map<CvDTreeNode*, GPC*> *leaf_map;

};
