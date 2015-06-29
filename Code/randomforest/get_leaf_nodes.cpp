#include "get_leaf_nodes.hpp"
#include <stdio.h>
#include <iostream>
#include <deque>

leaf_list& get_leaf_node( CvForestTree* tree )
{
    leaf_list* leaf_list = new ::leaf_list;
    std::deque<const CvDTreeNode*> right_nodes;
    const CvDTreeNode* node = tree->get_root();

    right_nodes.push_back(node);
	
    while (!right_nodes.empty()) {

	for (node = right_nodes.back(), right_nodes.pop_back(); node->left != NULL; node = node->left) {
	    right_nodes.push_back(node->right);
	}
	leaf_list->push_front(node);
    }
     
    return *leaf_list;
}
