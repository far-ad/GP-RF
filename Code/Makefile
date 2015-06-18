INCLUDES_OPENCV=-I/usr/include/opencv
INCLUDES_BOOST=-I/usr/include/boost 
INCL_GP=-I/usr/include/gp-lvm -I/usr/local/include/gp-lvm
INCLUDES=${INCLUDES_OPENCV} ${INCLUDES_BOOST} ${INCL_GP}

LIBS_OPENCV=`pkg-config --libs opencv`
LIBS_BOOST=-lboost_filesystem -lboost_system
LIBS_GP=-lgp -lgfortran -llapack -lblas

DBG_FLAGS=-ggdb

CC=g++ ${DBG_FLAGS}

all: gp rf

gp: gp/newGP_test

gp/newGP_test: gp/newGP_test.cpp gp/readTextData.cpp
	${CC} gp/newGP_test.cpp gp/readTextData.cpp ${INCL_GP} ${LIBS_GP} -o gp/newGP_test

rf: randomforest/test_randomforest randomforest/test_images randomforest/test_rgbd

randomforest/test_randomforest: randomforest/randomforest.o randomforest/test_randomforest.o
	${CC} randomforest/randomforest.o randomforest/test_randomforest.o ${LIBS_OPENCV} ${LIBS_BOOST} -o randomforest/test_randomforest

randomforest/test_images: randomforest/randomforest.o randomforest/test_images.o
	${CC} randomforest/randomforest.o randomforest/test_images.o ${LIBS_OPENCV} ${LIBS_BOOST} -o randomforest/test_images

randomforest/test_rgbd: randomforest/test_rgbd.o dataset/load_rgbd_cv.o randomforest/get_leaf_nodes.o
	${CC} dataset/load_rgbd_cv.o randomforest/test_rgbd.o randomforest/get_leaf_nodes.o ${LIBS_OPENCV} -o randomforest/test_rgbd

%.o: %.cpp
	${CC} ${INCLUDES} -c $< -o $@

clean:
	rm -f gp/newGP_test randomforest/test_{rgbd,images,randomforest} {dataset,gp,randomforest}/*.o

