#include <gp-lvm/CGp.h>

#include <iostream>
#include <gp-lvm/CMatrix.h>
#include <opencv/cv.h>

#include <fstream>
#include <iostream>

#include "writeTextData.hpp"
#include "cmat_read_rgbd_data.hpp"
#include "convert_Mat_to_CMatrix.hpp"
#include <fstream>



int main(int argc, char** argv) {
	std::cout << "GP running!" << std::endl;

	CMatrix *conTESTING, *conTRAINING;
	cv::Mat TESTING, TRAINING;
	const char* filenameTESTING= "/home/jack/Gaussian_Process/GPc/dataset/rgbdData/rgbdDataset-test.data";
	const char* filenameTRAINING= "/home/jack/Gaussian_Process/GPc/dataset/rgbdData/rgbdDataset-train.data";

	TESTING= cmat_read_rgbd_data( filenameTESTING );
	TRAINING= cmat_read_rgbd_data( filenameTRAINING );

	//100 elements for testing and training
	//testing case
	bool reducedSPACE=true; bool training=false; bool testing=true;
	writeTextData(TESTING, reducedSPACE, training, testing);
	//training case
	reducedSPACE=true; training=true; testing=false;
	writeTextData(TESTING, reducedSPACE, training, testing);


	//convert cv::Mat to CMatrix
/*	bool reducedSPACE=true;
	conTESTING=convert_Mat_to_CMatrix(TESTING, reducedSPACE);
	conTRAINING=convert_Mat_to_CMatrix(TRAINING, reducedSPACE);*/
//	TESTING.size;

//	TRAINING.




// CGp(CKern* kernel, CNoise* nois, CMatrix* Xin, int approxType=FTC, unsigned int actSetSize=0, int verbos=2);






	CGp *gp;
	gp= new CGp();
//	gp->X_u;
	std::cout << "done";
	return 0;

}

