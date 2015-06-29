INCLUDES_OPENCV=-I/usr/include/opencv
INCLUDES_BOOST=-I/usr/include/boost 
INCL_GP=-I/usr/include/gp-lvm -I/usr/local/include/gp-lvm
INCLUDES=${INCLUDES_OPENCV} ${INCLUDES_BOOST} ${INCL_GP}

LIBS_OPENCV=`pkg-config --libs opencv`
LIBS_BOOST=-lboost_filesystem -lboost_system
LIBS_GP=-lgp -lgfortran -llapack -lblas

DBG_FLAGS=-ggdb

CC=g++ ${DBG_FLAGS}

# general rule for building object files
%.o: %.cpp
	${CC} ${INCLUDES} -c $< -o $@

all: gp rf gprf

gp: gp/gpc gp/newGP_test gp/test_gp

gp/gpc: gp/test_gpc_cv.o gp/gpc.o dataset/load_rgbd_cv.o
	${CC} gp/test_gpc_cv.o gp/gpc.o dataset/load_rgbd_cv.o ${LIBS_GP} ${LIBS_OPENCV} -o gp/gpc 

gp/test_gp: gp/test_gp.o gp/read_rgbd_data.o
	${CC} gp/read_rgbd_data.o gp/test_gp.o ${LIBS_GP} -o gp/test_gp

gp/newGP_test: gp/newGP_test.cpp gp/read_rgbd_data.cpp
	${CC} gp/newGP_test.cpp gp/read_rgbd_data.cpp ${INCL_GP} ${LIBS_GP} -o gp/newGP_test

rf: randomforest/test_randomforest randomforest/test_images randomforest/test_rgbd

randomforest/test_randomforest: randomforest/randomforest.o randomforest/test_randomforest.o
	${CC} randomforest/randomforest.o randomforest/test_randomforest.o ${LIBS_OPENCV} ${LIBS_BOOST} -o randomforest/test_randomforest

randomforest/test_images: randomforest/randomforest.o randomforest/test_images.o
	${CC} randomforest/randomforest.o randomforest/test_images.o ${LIBS_OPENCV} ${LIBS_BOOST} -o randomforest/test_images

randomforest/test_rgbd: randomforest/test_rgbd.o dataset/load_rgbd_cv.o randomforest/get_leaf_nodes.o
	${CC} dataset/load_rgbd_cv.o randomforest/test_rgbd.o randomforest/get_leaf_nodes.o ${LIBS_OPENCV} -o randomforest/test_rgbd

gprf: gp/gpc randomforest/rfc.o GP_RF.o GP_RF_test.o
	${CC} gp/gpc.o randomforest/rfc.o GP_RF.o GP_RF_test.o ${LIBS_GP} ${LIBS_OPENCV} -o gprf

clean:
	rm -f {dataset,gp,randomforest,.}/*.o
	rm -f gprf gp/{test_gp,newGP_test,gpc} randomforest/test_{rgbd,images,randomforest} 

