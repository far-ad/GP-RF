#include <gp-lvm/CGp.h>
#include <gp-lvm/CIvm.h>
#include <gp-lvm/CMatrix.h>

#include "read_rgbd_data.hpp"

#include <fstream>
#include <iostream>
#include <cmath> // for 'abs(double)'


/**
 * Return a new label matrix that takes one label out of the others.
 *
 * The result will be a new matrix that has a 1 at every place where the given label occurred, otherwise -1.
 *
 * @param labels_matrix the matrix containing the original labels
 * @param filtered_label the label to pick
 * @return the new label matrix
 */
CMatrix *one_out_of_many(CMatrix *labels_matrix, double filtered_label)
{
	CMatrix *result = new CMatrix(*labels_matrix);

	for(int i=0; i<result->getRows(); i++) {
		for(int j=0; j<result->getCols(); j++) {
			if(abs(result->getVal(i,j) - filtered_label) <= FLT_EPSILON)
				result->setVal(1.0,i,j);
			else
				result->setVal(-1.0,i,j);
		}
	}
	
	return result;
}


int main(int argc, char** argv) {
	std::cout << "GP running!" << std::endl;

//	CMatrix *conTESTING, *conTESTING_labels, *conTRAINING, *conTRAINING_labels;
//	cv::Mat TESTING, TRAINING,TESTING_labels, TRAINING_labels;

	const char* filenameTRAINING= argv[1];
	const char* filenameTRAINING_labels= argv[2];
	const char* filenameTESTING= argv[3];
	const char* filenameTESTING_labels= argv[4];




//	TESTING= cmat_read_rgbd_data( filenameTESTING );
//	TRAINING= cmat_read_rgbd_data( filenameTRAINING );
//	TESTING_labels= cmat_read_rgbd_data( filenameTESTING_labels );
//	TRAINING_labels= cmat_read_rgbd_data( filenameTRAINING_labels );
//
//	//writing for each 100 elements for testing and training and for there labels
//	//testing and labels
//	bool reducedSPACE=true; bool training=false; bool testing=true;
//    bool training_labels=false; bool testing_labels=false;
//    //testing case
//	writeTextData(TESTING, reducedSPACE, training, testing, training_labels,testing_labels);
//	//testing labels
//	testing=false; testing_labels=true;
//	writeTextData(TESTING_labels, reducedSPACE, training, testing, training_labels,testing_labels);
//
//	//training case
//	reducedSPACE=true; training=true; testing=false; training_labels=false; testing_labels=false;
//	writeTextData(TESTING, reducedSPACE, training, testing, training_labels,testing_labels);
//	//testing labels
//	training=false; training_labels=true;
//	writeTextData(TRAINING_labels, reducedSPACE, training, testing, training_labels,testing_labels);
//





	//convert all elements from cv::Mat to CMatrix (use only when loading all data)
/*	bool reducedSPACE=true;
	conTESTING=convert_Mat_to_CMatrix(TESTING, reducedSPACE);
	conTESTING_labels=convert_Mat_to_CMatrix(TESTING_labels, reducedSPACE);
	conTRAINING=convert_Mat_to_CMatrix(TRAINING, reducedSPACE);
	conTRAINING_labels=convert_Mat_to_CMatrix(TRAINING_labels, reducedSPACE);*/

	// work only with #n_data instances for testing and training and train only on label 2
	int n_data = 40;
	int label = 2;

	// testing data
	CMatrix *conTESTING_100=read_rgbd_data_cmat( filenameTESTING, n_data);
	CMatrix *conTESTING_100_labels= one_out_of_many( read_rgbd_data_cmat( filenameTESTING_labels, n_data), label);

	// training data
	CMatrix *conTRAINING_100=read_rgbd_data_cmat( filenameTRAINING, n_data);
	CMatrix *conTRAINING_100_labels= one_out_of_many(read_rgbd_data_cmat( filenameTRAINING_labels, n_data), label);
//
//	  std::cout << "\nconTRAINING_100 number of rows : " << conTRAINING_100->getRows();
//	  std::cout << "\nconTRAINING_100 number of cols : " << conTRAINING_100->getCols();
//	  std::cout << "\nconTRAINING_100_labels number of rows : " << conTRAINING_100_labels->getRows();
//	  std::cout << "\nconTRAINING_100_labels number of cols : " << conTRAINING_100_labels->getCols();

	//  CIvm(CMatrix* inData, CMatrix* targetData,
   //   CKern* kernel, CNoise* noiseModel, int selectCrit,
   // unsigned int dVal, int verbos=2);


//calculation of the Kernel; selecting RBF Kernel
	CRbfKern  *Kernel_rbf = new CRbfKern( *conTRAINING_100 );
	std::cout << "Kernel_calculated value : " << Kernel_rbf->getInputDim() << endl;
	CDist* prior= new CGammaDist();
	prior->setParam(1.0,0);
	prior->setParam(1.0,1);
	Kernel_rbf->addPrior(prior,1);

//	  CMatrix();
//	//initilizing rbf kernel
//	  conTRAINING_100->trans();
//	CRbfKern  *Kernel_rbf= new CRbfKern(*conTRAINING_100  );
////	Kernel_rbf->compute(*conTRAINING_100, *conTRAINING_100);
////	Kernel_rbf->addPriorGrad(*conTRAINING_100);
//	Kernel_rbf->setVariance(1.3);
//	Kernel_rbf->setInverseWidth(0.3);
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
	std::cout << endl;
//
//
//	//initilizing white noise for classification
	CProbitNoise *noiseModel_classification= new CProbitNoise(conTRAINING_100_labels);//must be initilized conTRAINING_100

//
////	noiseModel_classification->setDefaultOptimiser(2);
//
//	noiseModel_classification->setVerbosity(1);
//	std::cout << "\n noiseModel_classification rbf getVerbosity : " << noiseModel_classification->getVerbosity();
//	std::cout << "\n noiseModel_classification rbf getVarSigma : " << noiseModel_classification->getVarSigma(1,1);
////	std::cout << "\n noiseModel_classification rbf getVarSigma : " << noiseModel_classification->setLearnRate(0.3);
//
//	noiseModel_classification->setDefaultOptimiser(2);
//	noiseModel_classification->setFuncEvalTerminate(true);
//	noiseModel_classification->setLearnRate(0.3);
//	noiseModel_classification->setMaxFuncEvals(2);
//	noiseModel_classification->setMomentum(0.5);
//	noiseModel_classification->setMus(1.5);
//	noiseModel_classification->setObjectiveTol(0.5);
//	noiseModel_classification->setParam(0.5,1);
//	noiseModel_classification->setParamTol(0.5);
//	noiseModel_classification->setVarSigmas(2);
//
//	std::cout << "\n noiseModel_classification getDefaultOptimiser : " << noiseModel_classification->getDefaultOptimiser();
//	std::cout << "\n noiseModel_classification getMaxFuncEvals : " << noiseModel_classification->getMaxFuncEvals();//getFuncEvalTerminate();
//	std::cout << "\n noiseModel_classification getLearnRate : " << noiseModel_classification->getLearnRate();
//	std::cout << "\n noiseModel_classification getMaxFuncEvals : " << noiseModel_classification->getMaxFuncEvals();
//	std::cout << "\n noiseModel_classification getMomentum : " << noiseModel_classification->getMomentum();
//	std::cout << "\n noiseModel_classification getMu : " << noiseModel_classification->getMu(1,1);//getMus();
//	std::cout << "\n noiseModel_classification getObjectiveTol : " << 	noiseModel_classification->getObjectiveTol();
//	std::cout << "\n noiseModel_classification getOptNumParams : " << noiseModel_classification->getOptNumParams();//getParam();
//	std::cout << "\n noiseModel_classification getParamTol : " << noiseModel_classification->getParamTol();
//	std::cout << "\n noiseModel_classification getVarSigma : " << noiseModel_classification->getVarSigma(1,1);//getVarSigmas(2);
//
//
//
//
//
//
//
//
//
//
////	std::cout << "\n Nois computeObjectiveVal : " << noiseModel_classification->computeObjectiveVal();
//
//
//
//
////	std::cout << "\n Nois setVerbosity : " << noiseModel_classification->setVerbosity(2);
//
//
//
//	//IVM classifier
//	unsigned int dVal=0; int verbos=2; int selectCrit=0;
//
//
//
//
//
////	CIvm *gp_classifier= new CIvm ();
////	CGp *new_gp= new CGp();
//
//
//
////	CMatrix* inData= new CMatrix(); CMatrix* targetData= new CMatrix();
////	unsigned int inDim=100;
////	CRbfKern kernel=CRbfKern(100);
////	CProbitNoise noiseModel=CProbitNoise();
////	CMatrix *OutputMatrix=new CMatrix();
////gplvm
	int selectCrit=CIvm::ENTROPY;
	int select_Active_Points=20;

	CIvm* gp_classifier= new CIvm (conTRAINING_100, conTRAINING_100_labels,
			Kernel_rbf,	noiseModel_classification, selectCrit,
			select_Active_Points, 3);

	std::cout << std::endl;
	std::cout << "==> optimizing model..." << std::endl;

	// optimise resets the values of the noise model and optimizes them for the given data
	gp_classifier->optimise();

	std::cout << std::endl;
	std::cout << "==> finished optimization" << std::endl;

//	std::cout << "\n logLikelihood : " << gp_classifier->logLikelihood() << endl;

	// test model using the test data
	std::cout << std::endl;
	std::cout << "==> testing model" << std::endl;
	gp_classifier->test(*conTESTING_100_labels, *conTESTING_100);

////	CMatrix *Pout=new CMatrix(100,14000);Pout->trans();
////	CIvm gp_classifier=CIvm();conTRAINING_100->trans();
////	gp_classifier.setVerbosity(3);
////	gp_classifier.init();
//	//gp_classifier.likelihoods(*Pout, *conTRAINING_100_labels,*conTRAINING_100);
//
////	CIvm *gp_classifier= new CIvm ();
////	gp_classifier->setDefaultOptimiser(2);
////	std::cout << "\n setting setDefaultOptimiser";
////	gp_classifier->setEpUpdate(true);
////	gp_classifier->setInputDim(1400);
////	std::cout << "\n setting setInputDim";
////	gp_classifier->setMomentum(1);
////	std::cout << "\n setting setMomentum";
////	gp_classifier->setVerbosity(2);
//
////	gp_classifier->setOutputDim(100);
////	std::cout << "\n setting setOutputDim";
////	gp_classifier->setVerbosity(2);
////	gp_classifier->getDirection(*OutputMatrix);conTRAINING_100
////	gp_classifier->setOutputDim(100);
////	CMatrix *testingValues=new CMatrix();
////	CNoise* testingNOISE=new CProbitNoise();
////	testingNOISE=gp_classifier->pnoise;
////	testingNOISE->getDirection(*testingValues);
//
////	(conTRAINING_100, conTRAINING_100_labels,
////				Kernel_rbf, noiseModel_classification,
////				selectCrit, dVal, verbos);
//
////	gp_classifier->init();

//	/*, int selectCrit,
//	   unsigned int dVal, int verbos=2);
//*/
//
//	std::fstream myfile(FILEname, std::ios_base::in);
//
//	    double a;
//	    int index=0;
//	    CMatrix *newTestmat= new CMatrix(1400,1400);
//	    newTestmat->getVal()
//	    while (myfile >> a)
//	    {
//
//	    	newTestmat[index,1]=a;
//
//	        printf("%f ", a);
//	        index++;
//	    }
//
//	    getchar();
//
//	    for( int i = 1; i < 20; i++)
//	    	   {
//	    		for( int j = 1; j < 20; j++){
//	    		std::cout << "\n value : " << newTestmat->getVal(j,i);
//	    		}
//	    	   }
//
//

//GP with IVM used for classification, outperforms SVM
//	CIvm *gp_classifier;
//	gp_classifier= new CIvm();
//	gp->X_u;

	std::cout << "done" << std::endl;

	return 0;

}



