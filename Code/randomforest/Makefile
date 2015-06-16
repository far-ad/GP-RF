LIBS_OPENCV=`pkg-config --libs opencv`
LIBS_BOOST=-lboost_filesystem -lboost_system
INCLUDES_OPENCV=-I/usr/include/opencv
INCLUDES_BOOST=-I/usr/include/boost 

DBG_FLAGS=-ggdb

CC=g++ ${DBG_FLAGS}

randomforest.o: randomforest.cpp
	${CC} randomforest.cpp ${INCLUDES_OPENCV} ${INCLUDES_BOOST} -c

test_randomforest.o: test_randomforest.cpp
	${CC} test_randomforest.cpp ${INCLUDES_OPENCV} ${INCLUDES_BOOST} -c

test_images.o: test_images.cpp
	${CC} test_images.cpp -c

test_randomforest: randomforest.o test_randomforest.o
	${CC} randomforest.o test_randomforest.o ${LIBS_OPENCV} ${LIBS_BOOST} -o test_randomforest

test_images: randomforest.o test_images.o
	${CC} randomforest.o test_images.o ${LIBS_OPENCV} ${LIBS_BOOST} -o test_images

test_rgbd: test_rgbd.cpp load_rgbd.cpp
	${CC} load_rgbd.cpp test_rgbd.cpp get_leaf_nodes.cpp ${LIBS_OPENCV} -o test_rgbd

clean:
	rm *.o test_rgbd test_images test_randomforest
