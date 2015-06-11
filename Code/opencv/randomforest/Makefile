LIBS_OPENCV=`pkg-config --libs opencv`\
LIBS_BOOST=-lboost_filesystem -lboost_system
INCLUDES_OPENCV=-I/usr/include/opencv
INCLUDES_BOOST=-I/usr/include/boost

randomforest.o: randomforest.cpp
	g++ randomforest.cpp ${INCLUDES_OPENCV} ${INCLUDES_BOOST} -c

test_randomforest.o: test_randomforest.cpp
	g++ test_randomforest.cpp ${INCLUDES_OPENCV} ${INCLUDES_BOOST} -c

test_images.o: test_images.cpp
	g++ test_images.cpp -c

test_randomforest: randomforest.o test_randomforest.o
	g++ randomforest.o test_randomforest.o ${LIBS_OPENCV} ${LIBS_BOOST} -o test_randomforest

test_images: randomforest.o test_images.o
	g++ randomforest.o test_images.o ${LIBS_OPENCV} ${LIBS_BOOST} -o test_images
