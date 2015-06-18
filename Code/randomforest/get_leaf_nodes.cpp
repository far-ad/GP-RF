#include <opencv/cv.h>
#include <opencv/ml.h>
#include <stdio.h>
#include <iostream>
#include <list>
#include <deque>

using namespace std;
using namespace cv;

list<const CvDTreeNode*> get_leaf_node( CvForestTree* tree )
{
	//const CvForestTree* tree = tree;
    std::list<const CvDTreeNode*> leaf_list;
    //std::stack<const CvDTreeNode*> right_nodes;
    std::deque<const CvDTreeNode*> right_nodes;
	const CvDTreeNode* node = tree->get_root();
	const CvDTreeNode* temp = node;
	const CvDTreeNode* tmp = node;
	right_nodes.push_back(node);
	
	while(!right_nodes.empty()) {

		for (node = right_nodes.back(), tmp = node; node->left != NULL; node = node->left) {
			
			right_nodes.push_back(node->right);
   		}
		temp = right_nodes.back();
        leaf_list.insert (leaf_list.begin(), temp);	
		leaf_list.insert (leaf_list.begin(), node);

		right_nodes.pop_back();//pop the element which is always on the end

		if (*right_nodes.back() == tmp) //pop the other element, but first need to find it
			right_nodes.pop_back(); 
		else if ((const CvDTreeNode*)*right_nodes.begin() == tmp)
				right_nodes.pop_front();
			else 
				printf("the element you want to delete is not at back nor at begin.");
	}
     
    return leaf_list;
}





