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
    std::deque<const CvDTreeNode*> test_nodes;//used to find the nodes that needed to be delete from right_nodes
    std::deque<const CvDTreeNode*> right_nodes;
	const CvDTreeNode* node = tree->get_root();
	const CvDTreeNode* temp = node;
	const CvDTreeNode* tmp = node;
	right_nodes.push_back(node);
	
	while (!right_nodes.empty()) {

		for (node = right_nodes.back(), tmp = node; node->left != NULL; node = node->left) {
			
			right_nodes.push_back(node->right);
   		}
		temp = right_nodes.back();
        leaf_list.insert (leaf_list.begin(), temp);	
		leaf_list.insert (leaf_list.begin(), node);

		right_nodes.pop_back();//pop the node which is always on the end

		if (right_nodes.back() == tmp) //pop the other node, but first need to find it
		{
			right_nodes.pop_back(); 
			printf("the node you deleted is at the back.\n");
		}
            
		else if (right_nodes.front() == tmp)
			{
				right_nodes.pop_front();
				printf("the node you deleted is at the front.\n");
			}
			else 
			{   
				while (right_nodes.front() != tmp)
				{
					test_nodes.push_front(right_nodes.front());
					right_nodes.pop_front();
				}
				right_nodes.pop_front();
				printf("the node you deleted is in the middle of deque.\n");
				while (!test_nodes.empty())
				{
					right_nodes.push_front(test_nodes.front());
					test_nodes.pop_front();
				}
			}
	}
     
    return leaf_list;
}





