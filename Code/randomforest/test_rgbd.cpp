#include <opencv/cv.h>
#include <opencv/ml.h>
#include <stdio.h>
#include <list>

#include "../dataset/load_rgbd_cv.h"

#define NUMBER_OF_CLASSES 5
#define NUMBER_OF_TRAINING_SAMPLES 200
#define NUMBER_OF_TESTING_SAMPLES 100

std::list<const CvDTreeNode*> get_leaf_node( CvForestTree* tree );


int main(int argc, char** argv)
{
  // std::cout<<FLT_EPSILON<<std::endl; 
  cv::Mat training_data, training_labels,testing_data, testing_labels;
  
  training_data = read_rgbd_data_cv(argv[1],NUMBER_OF_TRAINING_SAMPLES);
  training_labels = read_rgbd_data_cv(argv[2], NUMBER_OF_TRAINING_SAMPLES);
  testing_data = read_rgbd_data_cv(argv[3],NUMBER_OF_TESTING_SAMPLES);
  testing_labels = read_rgbd_data_cv(argv[4], NUMBER_OF_TESTING_SAMPLES);
  
 
  printf("dataset specs: %d samples with %d features\n", training_data.rows, training_data.cols);

  // define all the attributes as numerical
  // alternatives are CV_VAR_CATEGORICAL or CV_VAR_ORDERED(=CV_VAR_NUMERICAL)
  // that can be assigned on a per attribute basis

  cv::Mat var_type = cv::Mat(training_data.cols + 1, 1, CV_8U );
  var_type.setTo(cv::Scalar(CV_VAR_NUMERICAL) ); // all inputs are numerical
  var_type.at<uchar>(training_data.cols, 0) = CV_VAR_CATEGORICAL; // the labels are categorical

  /********************************步骤1：定义初始化Random Trees的参数******************************/
  float priors[] = {1,1,1,1,1};  // weights of each classification for classes
  CvRTParams params = CvRTParams(25, // max depth
				 50, // min sample count
				 0, // regression accuracy: N/A here
				 false, // compute surrogate split, no missing data
				 15, // max number of categories (use sub-optimal algorithm for larger numbers)
				 priors, // the array of priors
				 false,  // calculate variable importance
				 20,       // number of variables randomly selected at node and used to find the best split(s).
				 100,	 // max number of trees in the forest
				 0.01f,				// forrest accuracy
				 CV_TERMCRIT_ITER |	CV_TERMCRIT_EPS // termination cirteria
				 );
  
  /****************************步骤2：训练 Random Decision Forest(RDF)分类器*********************/
  // printf( "\nUsing training database: %s\n\n", argv[1]);
  CvRTrees* rtree = new CvRTrees;
  rtree->train(training_data, CV_ROW_SAMPLE, training_labels,
	       cv::Mat(), cv::Mat(), var_type, cv::Mat(), params);
  
  // perform classifier testing and report results
  cv::Mat test_sample;
  int correct_class = 0;
  int wrong_class = 0;
  int result;
  int label;
  int false_positives [NUMBER_OF_CLASSES] = {0,0,0,0,0};
  int false_negatives [NUMBER_OF_CLASSES] = {0,0,0,0,0};

  // printf( "\nUsing testing database: %s\n\n", argv[2]);

  for (int tsample = 0; tsample < testing_data.rows; tsample++)
    {	       
      // extract a row from the testing matrix
      test_sample = testing_data.row(tsample);
      // train on the testing data:
      // test_sample = training_data.row(tsample);
      /********************************步骤3：预测*********************************************/
      result = (int) rtree->predict(test_sample, cv::Mat());
      label  = (int) testing_labels.at<float>(tsample, 0);
      
      CvForestTree* tree = rtree->get_tree(1);
      std::list<const CvDTreeNode*> leaf_list;
      leaf_list = get_leaf_node( tree );
	  printf("Number of Leaf nodes: %ld", leaf_list.size());
      
      printf("Testing Sample %i -> class result (digit %d) - label (digit %d)\n", tsample, result, label);
      
      // if the prediction and the (true) testing classification are the same
      // (N.B. openCV uses a floating point decision tree implementation!)
      if (fabs(result - label)
	  >= FLT_EPSILON)
	{
	  // if they differ more than floating point error => wrong class
	  wrong_class++;
	  false_positives[(int) result]++;
	  false_negatives[(int) testing_labels.at<float>(tsample, 0)]++;
	}
      else
	{
	  // otherwise correct
	  correct_class++;
	}
    }

  printf( // "\nResults on the testing database: %s\n"
	 "\tCorrect classification: %d (%g%%)\n"
	 "\tWrong classifications: %d (%g%%)\n",
	 // argv[2],
	 correct_class, (double) correct_class*100/testing_data.rows,
	 wrong_class, (double) wrong_class*100/testing_data.rows);

  for (int i = 0; i < NUMBER_OF_CLASSES; i++)
    {
      printf( "\tClass (digit %d) false postives 	%d (%g%%)\n\t                false negatives  %d (%g%%)\n", i,
	      false_positives[i],
	      (double) false_positives[i]*100/testing_data.rows,
	      false_negatives[i],
	      (double) false_negatives[i]*100/testing_data.rows);
    }
        
  // all matrix memory free by destructors

  // all OK : main returns 0
  // result = rtree->predict(testing_data.row(79), cv::Mat());
  // float andi = result - testing_labels.at<float>(79, 0);
  // // std::cout<<training_labels.row(0).col(0)<<std::endl;
  // std::cout<<andi<<std::endl;
  return 0;
}
