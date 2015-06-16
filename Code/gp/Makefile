INCL_GP=-I/usr/include/gp-lvm -I/usr/local/include/gp-lvm
LIBS_GP=-lgp -lgfortran -llapack -lblas
CC=g++

all: newGP_test

newGP_test: newGP_test.cpp readTextData.cpp
	${CC} newGP_test.cpp readTextData.cpp ${INCL_GP} ${LIBS_GP} -o newGP_test

clean:
	rm newGP_test
