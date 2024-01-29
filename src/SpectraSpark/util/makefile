
CPP = g++
EIGEN =  $(shell pkg-config --cflags-only-I eigen3)
PYBIND = $(shell python3 -m pybind11 --includes)
PY_EXT = $(shell python3.11-config --extension-suffix)
CPPFLAGS = -Wall -std=c++23 -fPIC $(EIGEN)

all: util.o


test: test.out
	./test.out

test.out: test.cpp util.o
	$(CPP) $(CPPFLAGS) test.cpp util.o -o test.out

util.o: util.cpp util.hpp
	$(CPP) $(CPPFLAGS) -c util.cpp -o util.o