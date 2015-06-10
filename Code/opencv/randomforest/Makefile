LIBS=`pkg-config --libs opencv` -lboost_filesystem -lboost_system

randomforest: randomforest.cpp
	g++ randomforest.cpp -I/usr/include/opencv -I/usr/include/boost -o randomforest ${LIBS}
