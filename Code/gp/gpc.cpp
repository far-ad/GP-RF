#include "gpc.hpp"

#include <gp-lvm/CMatrix.h>

/**
 * Return a new label matrix that takes one label out of the others.
 *
 * The result will be a new matrix that has a 1 at every place where the given label occurred, otherwise -1.
 *
 * @param labels_matrix the matrix containing the original labels
 * @param filtered_label the label to pick
 * @return the new label matrix
 */
CMatrix *extract_label(CMatrix *labels_matrix, double filtered_label)
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



GPC::GPC(int n_features, double label) {
	input_dim = n_features;
	target_label = label;

	active_set_size = 20;
	select_crit = CIvm::ENTROPY;

	noise = new CProbitNoise();
	kernel = new CRbfKern( input_dim );

	CDist* prior = new CGammaDist();
	prior->setParam(1.0, 0);
	prior->setParam(1.0, 1);

	kernel->addPrior(prior,1);
}

void GPC::train(double *labels, double *features, int n_samples) {
	CMatrix* training_data = new CMatrix(n_samples, input_dim, features);
	CMatrix* training_labels = extract_label(new CMatrix(n_samples, 1, labels), target_label);

	// TODO:
	// Noise and kernel are just temporarily reinitialized here!
	// This should be removed as soon as possible to allow relearning or online learning
	noise = new CProbitNoise( training_labels );

	predictor = new CIvm(training_data, training_labels, kernel, noise, select_crit, active_set_size, 3);
}

double GPC::predict(double *features) {
	CMatrix ft(1, input_dim, features);
	CMatrix pred(1,1);

	predictor->out(pred, ft);
	return pred.getVal(0,0);
}

