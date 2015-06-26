#include "GP_RF.hpp"
#include "gpc.hpp"
#include <list>
#include "rfc.hpp"

using namespace std;

GP_RF::GP_RF(int n_features){

	this->n_features=n_features;
	leaf_map = new map<CvDTreeNode *, GPC>;
	this->forest_classifier= new RFC(n_features);


}

void GP_RF::train(double *training_labels, double *training_data, int n_samples){
	//train RF
	forest_classifier->train(training_labels, training_data, n_samples);

	// group the observations by leafs
	list<leaf_samples> subsets_per_leaf = forest_classifier->split_data_by_leafs(training_data, n_samples);

	//current leaf with number of data points
	for (list<leaf_samples>::iterator leafe_iter = subsets_per_leaf.begin(); leafe_iter < subsets_per_leaf.end(); ++leafe_iter) {
		double *leaf_labels=new double [leafe_iter->indices.size()];
		double *leaf_data=new double [leafe_iter->indices.size()*n_features];

		leaf_labels= leafe_iter->indices;
		int index_count=0;

		//goes through all data points belonging to the leaf
		for (list<int>::iterator  index_iter = leafe_iter->indices.begin(); index_iter < leafe_iter->indices.end(); ++index_iter) {
			leaf_labels[index_count] = training_labels[(*index_iter)];

			//going through all features of one line (observation)
			for (int i = 0;   i < n_features; ++i) {
				leaf_data[index_count * n_features + i]= training_data[(*index_iter) * n_features + i];
			}
			index_count++;
		}

		//for each leaf a GPC
		GPC *GP_classifier= new GPC(n_features);
		GP_classifier->train(leaf_labels,leaf_data,leafe_iter->indices.size());

		// store the trained classifier in a hash map for later use during testing
		leaf_map.insert( map<CvDTreeNode*, GPC*>::value_type(leafe_iter->leaf, GP_classifier) );
	}

	return;
}
