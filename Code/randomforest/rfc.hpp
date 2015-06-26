#include <opencv/cv.h>
#include <opencv/ml.h>
#include <stdio.h>
#include <list>
#include "../dataset/load_rgbd_cv.h"


typedef struct {
  CvDTreeNode* leaf;
  std::list<int> indices; 
} leaf_samples;

class RFC {
public:
	RFC();

	void train(cv::Mat training_data,cv::Mat training_labels);
	std::list<leaf_samples> split_data_by_leafs(cv::Mat training_data);
	std::list<CvDTreeNode*> get_leaf_list(cv::Mat testing_data);

private:
	CvRTParams params;
	CvRTrees* rtree;
}
