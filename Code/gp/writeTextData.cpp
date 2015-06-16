#include <opencv/cv.h>
#include <gp-lvm/CMatrix.h>
#include <fstream>
#include "writeTextData.hpp"

void writeTextData(cv::Mat inputDATA, bool reducedSPACE=true, bool training=false, bool testing=false, bool training_labels=false, bool testing_labels=false)
{

	cv::Mat B = cv::Mat(0,inputDATA.cols, CV_32FC1);
	//reducing original size for less computation time
	if(reducedSPACE){
		for (int i=0;i<100;i++){
			B.push_back(inputDATA.row(i));
		}
	 }else{
	 for(int i = 0; i<inputDATA.rows; i++){
	    B.push_back(inputDATA.row(i));
	 }
	 }

//stacking into an array
	 std::vector<double> array;
//cast must be float
	 array.assign((float*)B.datastart, (float*)B.dataend);


//writing the file
	 if(training){
	 std::ofstream output("100_elements_for_training.txt");
	 for (int i=0;i<array.size();i++)
	 {
		output << array[i] << " "; // behaves like cout - cout is also a stream

	 	output << "\n";
	 }
	 }
	 if(training_labels){
	 	 std::ofstream output("100_elements_for_training_labels.txt");
	 	 for (int i=0;i<array.size();i++)
	 	 {
	 		output << array[i] << " "; // behaves like cout - cout is also a stream

	 	 	output << "\n";
	 	 }
	 }
	 if(testing){
	 std::ofstream output("100_elements_for_testing.txt");
	 for (int i=0;i<array.size();i++)
	 {
	 	output << array[i] << " "; // behaves like cout - cout is also a stream

	 	output << "\n";
	 }
	 }
	 if(testing_labels){
	 	 std::ofstream output("100_elements_for_testing_labels.txt");
	 	 for (int i=0;i<array.size();i++)
	 	 {
	 	 	output << array[i] << " "; // behaves like cout - cout is also a stream

	 	 	output << "\n";
	 	 }
	 }


}
