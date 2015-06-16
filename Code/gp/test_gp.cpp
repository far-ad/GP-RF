#include <CGp.h>

#include <iostream>
#include <CMatrix.h>

int main(int argc, char** argv) {
	std::cout << "GP running!" << std::endl;

	CMatrix *TESTING, *TRAINING;
	const char* filenameTESTING= "adding training path";
	const char* filenameTRAINING= "adding testing path";

	TESTING= read_rgbd_data_cmat( filenameTESTING );
	TRAINING= read_rgbd_data_cmat( filenameTRAINING );

	CGp *gp = new CGp();

}

//loading training and testing sets
CMatrix *read_rgbd_data_cmat( const char* filename, int n_samples=0 )
{
	int rows;
	int cols;
	double *dataset = read_rgbd_data<double>( filename, &rows, &cols, n_samples );

	return new CMatrix(rows, cols, dataset);
}
