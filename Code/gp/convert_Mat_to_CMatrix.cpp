#include <opencv/cv.h>
#include <gp-lvm/CMatrix.h>
#include "convert_Mat_to_CMatrix.hpp"


CMatrix *convert_Mat_to_CMatrix(cv::Mat inputDATA, bool reducedSPACE) {

       cv::Mat B = cv::Mat(0,inputDATA.cols, CV_32FC1);
       //reducing original size for less computation time
       if(reducedSPACE){
    	   B.push_back(inputDATA.row(0));
    	   B.push_back(inputDATA.row(1));
    	   B.push_back(inputDATA.row(2));
       }else{
    	   for(int i = 0; i<inputDATA.rows; i++){
    		   B.push_back(inputDATA.row(i));
    	   }
       }

       //stacking into an array
       std::vector<double> array;
       //cast must be float
       array.assign((float*)B.datastart, (float*)B.dataend);

       //converting vector of doubles to CMatrix; usage for CGp(..., CMatrix* Xin, ...)
       CMatrix *m2= new CMatrix((unsigned int) B.rows,  (unsigned int) B.cols, array);
       return m2;
}

