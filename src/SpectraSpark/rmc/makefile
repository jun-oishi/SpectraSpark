
CPP = g++
EIGEN =  $(shell pkg-config --cflags-only-I eigen3)
PYBIND = $(shell python3 -m pybind11 --includes)
PY_EXT = $(shell python3.11-config --extension-suffix)
CPPFLAGS = -Wall -std=c++23 -fPIC $(EIGEN)
UTIL = ../util/util.o

all: rmc test

clean:
	rm -f *.o *$(PY_EXT) *.out

so: wrapper.cpp RmcCore.o rand.o PlanarRmc.o $(UTIL)
	$(CPP) $(CPPFLAGS) -shared $(PYBIND) wrapper.cpp RmcCore.o rand.o PlanarRmc.o $(UTIL) -o rmc$(PY_EXT)

testrun: test.out
	./test.out

test.out : test.o RmcCore.o rand.o PlanarRmc.o $(UTIL)
	$(CPP) $(CPPFLAGS) test.o RmcCore.o rand.o PlanarRmc.o $(UTIL) -o test.out

test.o : test.cpp
	$(CPP) $(CPPFLAGS) -c test.cpp -o test.o

RmcCore.o : RmcCore.cpp RmcCore.hpp rand.o
	$(CPP) $(CPPFLAGS) -c RmcCore.cpp -o RmcCore.o

rand.o : rand.cpp rand.hpp
	$(CPP) $(CPPFLAGS) -c rand.cpp -o rand.o

PlanarRmc.o: PlanarRmc.cpp PlanarRmc.hpp
	$(CPP) $(CPPFLAGS) -c PlanarRmc.cpp -o PlanarRmc.o
