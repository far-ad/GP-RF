#include <gp-lvm/CKern.h>
#include <gp-lvm/CNoise.h>
#include <gp-lvm/CIvm.h>

class GPC {
public:
	GPC(int n_features);

	void train(double* labels, double* features, int n_samples);
	double predict(double* features);
private:
	int input_dim;
	int active_set_size;
	int select_crit;

	CKern* kernel;
	CNoise* noise;
	CIvm* predictor;
};

