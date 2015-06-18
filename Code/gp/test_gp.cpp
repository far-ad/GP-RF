#include <gp-lvm/CGp.h>

#include <iostream>
#include <gp-lvm/CMatrix.h>
#include <opencv/cv.h>

#include <fstream>
#include <iostream>

#include "cmat_read_rgbd_data.hpp"
#include "convert_Mat_to_CMatrix.hpp"

int main(int argc, char** argv) {
	std::cout << "GP running!" << std::endl;

	CMatrix *conTESTING, *conTRAINING;
	cv::Mat TESTING, TRAINING;
	const char* filenameTESTING= "/home/jack/Gaussian_Process/GPc/dataset/rgbdData/rgbdDataset-test.data";
	const char* filenameTRAINING= "/home/jack/Gaussian_Process/GPc/dataset/rgbdData/rgbdDataset-train.data";

	TESTING= cmat_read_rgbd_data( filenameTESTING );
	TRAINING= cmat_read_rgbd_data( filenameTRAINING );

	//convert cv::Mat to CMatrix
	bool reducedSPACE=true;
	conTESTING=convert_Mat_to_CMatrix(TESTING, reducedSPACE);
	conTRAINING=convert_Mat_to_CMatrix(TRAINING, reducedSPACE);
//	TESTING.size;

//	TRAINING.
// CGp(CKern* kernel, CNoise* nois, CMatrix* Xin, int approxType=FTC, unsigned int actSetSize=0, int verbos=2);

	CGp *gp;
	gp= new CGp();
//	gp->X_u;
	std::cout << "done";
	return 0;

}

//loading training and testing sets
CMatrix *read_rgbd_data_cmat( const char* filename, int n_samples=0 )
{
	int rows;
	int cols;
	double *dataset = read_rgbd_data<double>( filename, &rows, &cols, n_samples );

	return new CMatrix(rows, cols, dataset);
}
