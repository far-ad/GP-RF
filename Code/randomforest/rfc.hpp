#ifndef RFC_HPP
#define RFC_HPP

#include <opencv/cv.h>
#include <opencv/ml.h>
#include <stdio.h>
#include <list>
#include <map>
#include "../dataset/load_rgbd_cv.h"


typedef struct {
public:
  CvDTreeNode* leaf;
  std::list<int> indices;
} leaf_samples;

typedef std::map<CvDTreeNode*, std::list<int> > leaf_map;


class RFC {
public:
	RFC(int n_features);

	void train(double* training_labels, double* training_data, int n_samples);
	leaf_map* split_data_by_leafs(double* training_data, int n_samples);
	std::list<CvDTreeNode*> get_leaf_list(double* testing_data, int n_samples);

private:
	CvRTParams params;
	CvRTrees* rtree;

	int n_features;

	cv::Mat dvec2dataMat(double* vec, int n_features);
	cv::Mat dvec2labelsMat(double*vec, int n_features);
};

#endif
