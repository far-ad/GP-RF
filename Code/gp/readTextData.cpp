#include <opencv/cv.h>
#include <iostream>
#include <gp-lvm/CMatrix.h>
#include <fstream>
#include "readTextData.hpp"



CMatrix *readTextData(bool training=false, bool testing=false, char *FILEname="/home/jack/Desktop/Project/Final/GP-RF/Code/gp/100_elements_for_training.txt")
{
	int samples=100, attributes=14000;
	if(training){
	//read data from text file
	std::vector<double> data(samples*attributes);
	std::ifstream file(FILEname);

	for (int i = 0; i < samples*attributes; ++i) {
		file >> data[i];
	}
	return new CMatrix((unsigned int) samples,  (unsigned int) attributes, data);
	}


	if(testing){
		//read data from text file
		std::vector<double> data(samples*attributes);
		std::ifstream file(FILEname);

		for (int i = 0; i < samples*attributes; ++i) {
			file >> data[i];
		}
		return new CMatrix((unsigned int) samples,  (unsigned int) attributes, data);
		}
	return 0;
}
