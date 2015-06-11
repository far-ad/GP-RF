#include <opencv/cv.h>
#include <opencv/ml.h>

cv::Mat read_rgbd_data( const char* filename, int n_samples );

int main(int argc, char** argv)
{
  cv::Mat training_data, training_labels;
  training_data = read_rgbd_data(argv[1],10);
  training_labels = read_rgbd_data(argv[2], 10);
 
  // define all the attributes as numerical
  // alternatives are CV_VAR_CATEGORICAL or CV_VAR_ORDERED(=CV_VAR_NUMERICAL)
  // that can be assigned on a per attribute basis

  cv::Mat var_type = cv::Mat(training_data.cols + 1, 1, CV_8U );
  var_type.setTo(cv::Scalar(CV_VAR_NUMERICAL) ); // all inputs are numerical

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
  
}  
