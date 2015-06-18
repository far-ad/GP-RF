#include <gp-lvm/CGp.h>
#include <gp-lvm/CIvm.h>

#include <iostream>
#include <gp-lvm/CMatrix.h>
#include <opencv/cv.h>
#include <gp-lvm/ivm.h>
#include <fstream>
#include <iostream>

#include "writeTextData.hpp"
#include "cmat_read_rgbd_data.hpp"
#include "convert_Mat_to_CMatrix.hpp"
#include "readTextData.hpp"
#include <fstream>


#include "gp-lvm/COptimisable.h"


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

	  std::cout << "\nconTRAINING_100 number of rows : " << conTRAINING_100->getRows();
	  std::cout << "\nconTRAINING_100 number of cols : " << conTRAINING_100->getCols();
	  std::cout << "\nconTRAINING_100_labels number of rows : " << conTRAINING_100_labels->getRows();
	  std::cout << "\nconTRAINING_100_labels number of cols : " << conTRAINING_100_labels->getCols();

	//  CIvm(CMatrix* inData, CMatrix* targetData,
   //   CKern* kernel, CNoise* noiseModel, int selectCrit,
   // unsigned int dVal, int verbos=2);


//calculation of the Kernel; selecting RBF Kernel
//	CRbfKern  *Kernel_calculated = new CRbfKern( *conTRAINING_100 );

	//initilizing rbf kernel
	  conTRAINING_100->trans();
	CRbfKern  *Kernel_rbf= new CRbfKern(14000  );
//	Kernel_rbf->compute(*conTRAINING_100, *conTRAINING_100);
//	Kernel_rbf->addPriorGrad(*conTRAINING_100);
	Kernel_rbf->setVariance(1.3);
	Kernel_rbf->setInverseWidth(0.3);
	std::cout << "\n Kernel rbf CurrentVersion : " << Kernel_rbf->getCurrentVersion();
	std::cout << "\n Kernel rbf BaseType : " << Kernel_rbf->getBaseType();
	std::cout << "\n Kernel rbf InputDim : " << Kernel_rbf->getInputDim();
	std::cout << "\n Kernel rbf getInverseWidth : " << Kernel_rbf->getInverseWidth();
	std::cout << "\n Kernel rbf getLengthScale : " << Kernel_rbf->getLengthScale();
	std::cout << "\n Kernel rbf getMinCompatVersion : " << Kernel_rbf->getMinCompatVersion();
	std::cout << "\n Kernel rbf getName : " << Kernel_rbf->getName();
	std::cout << "\n Kernel rbf getNumParams : " << Kernel_rbf->getNumParams();
	std::cout << "\n Kernel rbf getNumPriors : " << Kernel_rbf->getNumPriors();
	std::cout << "\n Kernel rbf getNumTransforms : " << Kernel_rbf->getNumTransforms();
	std::cout << "\n Kernel rbf getVariance : " << Kernel_rbf->getVariance();
	std::cout << "\n Kernel rbf getWhite : " << Kernel_rbf->getWhite();


	//initilizing white noise for classification
	CProbitNoise *noiseModel_classification= new CProbitNoise(conTRAINING_100);//must be initilized conTRAINING_100


//	noiseModel_classification->setDefaultOptimiser(2);

	noiseModel_classification->setVarSigmas(2);
	noiseModel_classification->setVerbosity(1);
	std::cout << "\n noiseModel_classification rbf getVerbosity : " << noiseModel_classification->getVerbosity();
	std::cout << "\n noiseModel_classification rbf getVarSigma : " << noiseModel_classification->getVarSigma(1,1);
//	std::cout << "\n noiseModel_classification rbf getVarSigma : " << noiseModel_classification->setLearnRate(0.3);

	noiseModel_classification->setDefaultOptimiser(2);
	noiseModel_classification->setFuncEvalTerminate(true);
	noiseModel_classification->setLearnRate(0.3);
	noiseModel_classification->setMaxFuncEvals(2);
	noiseModel_classification->setMomentum(0.5);
	noiseModel_classification->setMus(1.5);
	noiseModel_classification->setObjectiveTol(0.5);
	noiseModel_classification->setParam(0.5,1);
	noiseModel_classification->setParamTol(0.5);
	noiseModel_classification->setVarSigmas(2);

	std::cout << "\n noiseModel_classification getDefaultOptimiser : " << noiseModel_classification->getDefaultOptimiser();
	std::cout << "\n noiseModel_classification getMaxFuncEvals : " << noiseModel_classification->getMaxFuncEvals();//getFuncEvalTerminate();
	std::cout << "\n noiseModel_classification getLearnRate : " << noiseModel_classification->getLearnRate();
	std::cout << "\n noiseModel_classification getMaxFuncEvals : " << noiseModel_classification->getMaxFuncEvals();
	std::cout << "\n noiseModel_classification getMomentum : " << noiseModel_classification->getMomentum();
	std::cout << "\n noiseModel_classification getMu : " << noiseModel_classification->getMu(1,1);//getMus();
	std::cout << "\n noiseModel_classification getObjectiveTol : " << 	noiseModel_classification->getObjectiveTol();
	std::cout << "\n noiseModel_classification getOptNumParams : " << noiseModel_classification->getOptNumParams();//getParam();
	std::cout << "\n noiseModel_classification getParamTol : " << noiseModel_classification->getParamTol();
	std::cout << "\n noiseModel_classification getVarSigma : " << noiseModel_classification->getVarSigma(1,1);//getVarSigmas(2);










//	std::cout << "\n Nois computeObjectiveVal : " << noiseModel_classification->computeObjectiveVal();




//	std::cout << "\n Nois setVerbosity : " << noiseModel_classification->setVerbosity(2);



	//IVM classifier
	unsigned int dVal=1; int verbos=2; int selectCrit=0;





//	CIvm *gp_classifier= new CIvm ();
//	CGp *new_gp= new CGp();

	conTRAINING_100






	CIvm *gp_classifier= new CIvm (conTRAINING_100, conTRAINING_100_labels,
			Kernel_rbf, noiseModel_classification,
			selectCrit, dVal, verbos);
//	gp_classifier->init();

//	std::cout << "conTRAINING_100_labels number of cols : " << gp_classifier->logLikelihood();

	/*, int selectCrit,
	   unsigned int dVal, int verbos=2);
*/







//GP with IVM used for classification, outperforms SVM
//	CIvm *gp_classifier;
//	gp_classifier= new CIvm();
//	gp->X_u;



	std::cout << "\n done";


	return 0;

}

