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
	return cv::Mat( n_samples, n_features, CV_32FC1, vec );
}

/**
 * Converts a vector of labels to a cv matrix.
 */
cv::Mat RFC::dvec2labelsMat(double* vec, int n_samples) {
	return cv::Mat( n_samples, 1, CV_32FC1, vec );
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


std::list<leaf_samples> RFC::split_data_by_leafs(double* training_data_d, int n_samples) {
	cv::Mat training_data = dvec2dataMat(training_data_d, n_samples);

CvDTreeNode* leaf_nodes [training_data.rows*NUMBER_OF_TREES];

  for (int i = 0; i < NUMBER_OF_TREES; i++)
    {
	  CvForestTree* tree = rtree->get_tree(i);
	  for (int tsample = 0; tsample < training_data.rows; tsample++)
		{
		    cv::Mat train_sample = training_data.row(tsample);
      		CvDTreeNode* leaf_node = tree->predict(train_sample, cv::Mat());
      		leaf_nodes[tsample*i+tsample] = leaf_node;
		}
    }


  std::list<leaf_samples> leaf_with_samples;
  for (int i = 0; i < training_data.rows*NUMBER_OF_TREES; i++)
    {
      CvDTreeNode* leaf_node = leaf_nodes[i];

      if (leaf_node != NULL)
	  {
		leaf_samples leaf_sample;
		leaf_sample.leaf = leaf_node;
		leaf_sample.indices.push_front(i);
		printf("\nValue of leaf: %f\n", leaf_node->value);
		printf("Smaple indices for leaf:\n");
		printf(" %d", i);

		for (int j=i+1; j < training_data.rows*NUMBER_OF_TREES; j++)
	  	{
	    	if (leaf_node == leaf_nodes[j])
			{
	      		leaf_sample.indices.push_front(j);
	      		printf(" %lu", j);
	      		leaf_nodes[j] = NULL;
	    	}
	  	}
		leaf_with_samples.push_front(leaf_sample);
      }
    }
	return leaf_with_samples;
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

