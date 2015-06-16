#include <opencv/cv.h>
#include <opencv/ml.h>
#include <stdio.h>
#include <iostream>
#include <list>
#include <stack>

using namespace std;
using namespace cv;

list<const CvDTreeNode*> get_leaf_node( CvForestTree* tree )
{
	//const CvForestTree* tree = tree;
    std::list<const CvDTreeNode*> leaf_list;
    std::stack<const CvDTreeNode*> right_nodes;

	const CvDTreeNode* node = tree->get_root();
	right_nodes.push(node);
	 
	while(!right_nodes.empty()) {

		for (node = right_nodes.top(); node->left != NULL; node = node->left) {
			right_nodes.pop();
			right_nodes.push(node->right);
   		}	

		leaf_list.insert (leaf_list.begin(), node);
	}
     
    return leaf_list;
}





