#include <opencv/cv.h>
#include <opencv/ml.h>
#include <list>

using namespace std;
using namespace cv;

typedef list<const CvDTreeNode*> leaf_list;

leaf_list& get_leaf_node( CvForestTree* tree );
