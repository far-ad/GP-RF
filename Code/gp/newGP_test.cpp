#include <gp-lvm/CGp.h>
#include <gp-lvm/CIvm.h>

#include <iostream>
#include <gp-lvm/CMatrix.h>
#include <opencv/cv.h>

#include <fstream>
#include <iostream>

#include "writeTextData.hpp"
#include "cmat_read_rgbd_data.hpp"
#include "convert_Mat_to_CMatrix.hpp"
#include "readTextData.hpp"
#include <fstream>



int main(int argc, char** argv) {
	std::cout << "GP running!" << std::endl;
/*
	CMatrix *conTESTING, *conTESTING_labels, *conTRAINING, *conTRAINING_labels;
	cv::Mat TESTING, TRAINING,TESTING_labels, TRAINING_labels;
	const char* filenameTESTING= "/home/jack/Gaussian_Process/GPc/dataset/rgbdData/rgbdDataset-test.data";
	const char* filenameTRAINING= "/home/jack/Gaussian_Process/GPc/dataset/rgbdData/rgbdDataset-train.data";
	const char* filenameTESTING_labels= "/home/jack/Gaussian_Process/GPc/dataset/rgbdData/rgbdDataset-test.labels";
	const char* filenameTRAINING_labels= "/home/jack/Gaussian_Process/GPc/dataset/rgbdData/rgbdDataset-train.labels";
	TESTING= cmat_read_rgbd_data( filenameTESTING );
	TRAINING= cmat_read_rgbd_data( filenameTRAINING );
	TESTING_labels= cmat_read_rgbd_data( filenameTESTING_labels );
	TRAINING_labels= cmat_read_rgbd_data( filenameTRAINING_labels );

	//writing for each 100 elements for testing and training and for there labels
	//testing and labels
	bool reducedSPACE=true; bool training=false; bool testing=true;
    bool training_labels=false; bool testing_labels=false;
    //testing case
	writeTextData(TESTING, reducedSPACE, training, testing, training_labels,testing_labels);
	//testing labels
	testing=false; testing_labels=true;
	writeTextData(TESTING_labels, reducedSPACE, training, testing, training_labels,testing_labels);

	//training case
	reducedSPACE=true; training=true; testing=false; training_labels=false; testing_labels=false;
	writeTextData(TESTING, reducedSPACE, training, testing, training_labels,testing_labels);
	//testing labels
	training=false; training_labels=true;
	writeTextData(TRAINING_labels, reducedSPACE, training, testing, training_labels,testing_labels);

*/




	//convert all elements from cv::Mat to CMatrix (use only when loading all data)
/*	bool reducedSPACE=true;
	conTESTING=convert_Mat_to_CMatrix(TESTING, reducedSPACE);
	conTESTING_labels=convert_Mat_to_CMatrix(TESTING_labels, reducedSPACE);
	conTRAINING=convert_Mat_to_CMatrix(TRAINING, reducedSPACE);
	conTRAINING_labels=convert_Mat_to_CMatrix(TRAINING_labels, reducedSPACE);*/


	//work only with 100 instances for testing and training
	//testing


	bool training=false, testing=true;
	char *FILEname="/home/jack/Desktop/Project/Final/GP-RF/Code/gp/100_elements_for_testing.txt";
	char *FILEname_labels="/home/jack/Desktop/Project/Final/GP-RF/Code/gp/100_elements_for_testing_labels.txt";
	CMatrix *conTESTING_100=readTextData( training,  testing, FILEname);
	CMatrix *conTESTING_100_labels=readTextData( training,  testing, FILEname_labels);
	//training
	training=true, testing=false;
	FILEname="/home/jack/Desktop/Project/Final/GP-RF/Code/gp/100_elements_for_training.txt";
	FILEname_labels="/home/jack/Desktop/Project/Final/GP-RF/Code/gp/100_elements_for_training_labels.txt";
	CMatrix *conTRAINING_100=readTextData( training,  testing, FILEname);
	CMatrix *conTRAINING_100_labels=readTextData( training,  testing, FILEname_labels);

	  std::cout << "conTRAINING_100 number of rows : " << conTRAINING_100->getRows();
	  std::cout << "\n "  << endl;
	  std::cout << "conTRAINING_100 number of cols : " << conTRAINING_100->getCols();
	  std::cout << "\n "  << endl;
	  std::cout << "conTRAINING_100_labels number of rows : " << conTRAINING_100_labels->getRows();
	  std::cout << "\n "  << endl;
	  std::cout << "conTRAINING_100_labels number of cols : " << conTRAINING_100_labels->getCols();
	  std::cout << "\n "  << endl;

	//  CIvm(CMatrix* inData, CMatrix* targetData,
   //   CKern* kernel, CNoise* noiseModel, int selectCrit,
   // unsigned int dVal, int verbos=2);


//calculation of the Kernel; selecting RBF Kernel
//	CRbfKern  *Kernel_calculated = new CRbfKern( *conTRAINING_100 );

	//initilizing rbf kernel
	CRbfKern  *Kernel_rbf= new CRbfKern( );
	//initilizing white noise for classification
	CProbitNoise *noiseModel_classification= new CProbitNoise();

	//IVM classifier
	unsigned int dVal=1; int verbos=1; int selectCrit=1;
	CIvm *gp_classifier= new CIvm ();
//	CGp *new_gp= new CGp();
	/*
	CIvm *gp_classifier= new CIvm (conTRAINING_100, conTRAINING_100_labels,
			Kernel_rbf, noiseModel_classification,
			selectCrit, dVal, verbos);
			*/
	/*, int selectCrit,
	   unsigned int dVal, int verbos=2);
*/







//GP with IVM used for classification, outperforms SVM
//	CIvm *gp_classifier;
//	gp_classifier= new CIvm();
//	gp->X_u;



	std::cout << "done";


	return 0;

}

