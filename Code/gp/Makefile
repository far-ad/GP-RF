INCL_GP=-I/usr/include/gp-lvm -I/usr/local/include/gp-lvm
LIBS_GP=-lgp -lgfortran -llapack -lblas
CC=g++

all: test_gp

test_gp: test_gp.cpp
	${CC} test_gp.cpp ${INCL_GP} ${LIBS_GP} -o test_gp

clean:
	rm test_gp
