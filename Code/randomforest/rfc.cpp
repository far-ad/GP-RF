#include "rfc.hpp"

#define NUMBER_OF_TREES 100


void dvec2fvec(double* vec, float* res_vec, int n)
{
	for(int i=0; i<n; i++) {
		res_vec[i] = (float) vec[i];
	}
}

/**
 * Converts vector of observations to a cv matrix.
 */
cv::Mat RFC::dvec2dataMat(double* vec, int n_samples) {
  float* data_f = new float[n_features*n_samples];
	dvec2fvec(vec, data_f, n_features*n_samples);

	return cv::Mat( n_samples, n_features, CV_32FC1, data_f );
}

/**
 * Converts a vector of labels to a cv matrix.
 */
cv::Mat RFC::dvec2labelsMat(double* vec, int n_samples) {
	float* data_f = new float[n_features*n_samples];
	dvec2fvec(vec, data_f, n_samples);

	return cv::Mat( n_samples, 1, CV_32FC1, data_f );
}


RFC::RFC(int n_features) {
	this->n_features = n_features;

	float priors[] = {1,1,1,1,1};  // weights of each classification for classes
	params = CvRTParams(25, // max depth
		50, // min sample count
		0, // regression accuracy: N/A here
		false, // compute surrogate split, no missing data
		15, // max number of categories (use sub-optimal algorithm for larger numbers)
		priors, // the array of priors
		false,  // calculate variable importance
		20,       // number of variables randomly selected at node and used to find the best split(s).
		NUMBER_OF_TREES,	 // max number of trees in the forest
		0.01f,				// forrest accuracy
		CV_TERMCRIT_ITER |	CV_TERMCRIT_EPS // termination cirteria
		);
}


void RFC::train(double* training_labels_d, double* training_data_d, int n_samples) {

	cv::Mat training_labels = dvec2labelsMat(training_labels_d, n_samples);
	cv::Mat training_data = dvec2dataMat(training_data_d, n_samples);

cv::Mat var_type = cv::Mat(training_data.cols + 1, 1, CV_8U );
  var_type.setTo(cv::Scalar(CV_VAR_NUMERICAL) ); // all inputs are numerical
  var_type.at<uchar>(training_data.cols, 0) = CV_VAR_CATEGORICAL; // the labels are categorical


rtree = new CvRTrees;
  rtree->train(training_data, CV_ROW_SAMPLE, training_labels,
	       cv::Mat(), cv::Mat(), var_type, cv::Mat(), params);
}


leaf_map* RFC::split_data_by_leafs(double* training_data_d, int n_samples) {
  cv::Mat training_data = dvec2dataMat(training_data_d, n_samples);

  // map leafs to lists of indices of observations
  leaf_map* leafs_indices_map = new leaf_map;

  int n_trees = rtree->get_tree_count();

  for (int i_sample = 0; i_sample < training_data.rows; i_sample++)
  {
	  cv::Mat sample = training_data.row(i_sample);

	  for (int i_tree = 0; i_tree < n_trees; i_tree++)
	  {
		  CvDTreeNode* leaf_node = rtree->get_tree(i_tree)->predict(sample, cv::Mat());
		  if( leafs_indices_map->find(leaf_node) != leafs_indices_map->end() ) {
			  std::list<int> l = (*leafs_indices_map)[leaf_node];
			  l.insert(l.end(), i_sample);
		  }
		  else {
			  leafs_indices_map->insert(leaf_map::value_type(leaf_node, std::list<int>(1, i_sample)));
		  }
	  }
  }

  return leafs_indices_map;
}

std::list<CvDTreeNode*>  RFC::get_leaf_list(double* testing_data_d, int n_samples)
{
	cv::Mat testing_data = dvec2dataMat(testing_data_d, n_samples);

	std::list<CvDTreeNode*> leaf_list;
	for (int i = 0; i < NUMBER_OF_TREES; i++)
	{
		CvForestTree* tree = rtree->get_tree(i);
		CvDTreeNode* leaf_node = tree->predict(testing_data, cv::Mat());
		leaf_list.push_front(leaf_node);
	}
	return leaf_list;
}

