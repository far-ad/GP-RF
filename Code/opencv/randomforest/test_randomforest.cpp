#include <cv.h>       // opencv general include file
#include <ml.h>		  // opencv machine learning include file
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>

#include <iostream>
#include <fstream>
#include <string>

#include <vector>

using namespace std;
using namespace cv;

/******************************************************************************/

int main( int argc, char** argv )
{
    
	for (int i=0; i< argc; i++)
		std::cout<<argv[i]<<std::endl;
	
	
	// lets just check the version first
	printf ("OpenCV version %s (%d.%d.%d)\n",
            CV_VERSION,
            CV_MAJOR_VERSION, CV_MINOR_VERSION, CV_SUBMINOR_VERSION);
	
	if(argc < 5) {
		printf("please provide training images, training labels, testing images and testing labels as arguments!!!\n");
		return 1;
	}
	
	//定义训练数据与标签矩阵
    Mat training_data = Mat(NUMBER_OF_TRAINING_SAMPLES, ATTRIBUTES_PER_SAMPLE, CV_32FC1);
    Mat training_classifications = Mat(NUMBER_OF_TRAINING_SAMPLES, 1, CV_32FC1);

    //定义测试数据矩阵与标签
    Mat testing_data = Mat(NUMBER_OF_TESTING_SAMPLES, ATTRIBUTES_PER_SAMPLE, CV_32FC1);
    Mat testing_classifications = Mat(NUMBER_OF_TESTING_SAMPLES, 1, CV_32FC1);

    // define all the attributes as numerical
    // alternatives are CV_VAR_CATEGORICAL or CV_VAR_ORDERED(=CV_VAR_NUMERICAL)
    // that can be assigned on a per attribute basis

    Mat var_type = Mat(ATTRIBUTES_PER_SAMPLE + 1, 1, CV_8U );
    var_type.setTo(Scalar(CV_VAR_NUMERICAL) ); // all inputs are numerical

    // this is a classification problem (i.e. predict a discrete number of class
    // outputs) so reset the last (+1) output var_type element to CV_VAR_CATEGORICAL

    var_type.at<uchar>(ATTRIBUTES_PER_SAMPLE, 0) = CV_VAR_CATEGORICAL;

    double result; // value returned from a prediction

	/* for loading csv
    //加载训练数据集和测试数据集
    if (read_data_from_csv(argv[1], training_data, training_classifications, NUMBER_OF_TRAINING_SAMPLES) &&
            read_data_from_csv(argv[2], testing_data, testing_classifications, NUMBER_OF_TESTING_SAMPLES))
    */
    // loading raw data
    if (read_mnist_images(argv[1], training_data, NUMBER_OF_TRAINING_SAMPLES) &&
    		read_mnist_labels(argv[2], training_classifications, NUMBER_OF_TRAINING_SAMPLES) &&
    		read_mnist_images(argv[3], testing_data, NUMBER_OF_TESTING_SAMPLES) &&
    		read_mnist_labels(argv[4], testing_classifications, NUMBER_OF_TESTING_SAMPLES))
    {
      /********************************步骤1：定义初始化Random Trees的参数******************************/
        float priors[] = {1,1,1,1,1,1,1,1,1,1};  // weights of each classification for classes
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
        rtree->train(training_data, CV_ROW_SAMPLE, training_classifications,
                     Mat(), Mat(), var_type, Mat(), params);

        // perform classifier testing and report results
        Mat test_sample;
        int correct_class = 0;
        int wrong_class = 0;
        int false_positives [NUMBER_OF_CLASSES] = {0,0,0,0,0,0,0,0,0,0};
        int false_negatives [NUMBER_OF_CLASSES] = {0,0,0,0,0,0,0,0,0,0};

        // printf( "\nUsing testing database: %s\n\n", argv[2]);

        for (int tsample = 0; tsample < NUMBER_OF_TESTING_SAMPLES; tsample++)
        {

            // extract a row from the testing matrix
            test_sample = testing_data.row(tsample);
            // train on the testing data:
            // test_sample = training_data.row(tsample);
        /********************************步骤3：预测*********************************************/
            result = rtree->predict(test_sample, Mat());

            printf("Testing Sample %i -> class result (digit %d)\n", tsample, (int) result);

            // if the prediction and the (true) testing classification are the same
            // (N.B. openCV uses a floating point decision tree implementation!)
            if (fabs(result - testing_classifications.at<float>(tsample, 0))
                    >= FLT_EPSILON)
            {
                // if they differ more than floating point error => wrong class
                wrong_class++;
                false_positives[(int) result]++;
                false_negatives[(int) testing_classifications.at<float>(tsample, 0)]++;
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
                correct_class, (double) correct_class*100/NUMBER_OF_TESTING_SAMPLES,
                wrong_class, (double) wrong_class*100/NUMBER_OF_TESTING_SAMPLES);

        for (int i = 0; i < NUMBER_OF_CLASSES; i++)
        {
            printf( "\tClass (digit %d) false postives 	%d (%g%%)\n\t                false negatives  %d (%g%%)\n", i,
                    false_positives[i],
                    (double) false_positives[i]*100/NUMBER_OF_TESTING_SAMPLES,
                    false_negatives[i],
                    (double) false_negatives[i]*100/NUMBER_OF_TESTING_SAMPLES);
        }
        
        // all matrix memory free by destructors

        // all OK : main returns 0
        return 0;
    }

    // not OK : main returns -1
    return -1;
}
/******************************************************************************/

