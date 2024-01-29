
CPP = g++
EIGEN =  $(shell pkg-config --cflags-only-I eigen3)
PYBIND = $(shell python3 -m pybind11 --includes)
PY_EXT = $(shell python3.11-config --extension-suffix)
CPPFLAGS = -Wall -std=c++23 -fPIC $(EIGEN)
SRCDIR = src/SpectraSpark

RMCSOURCES := $(shell ls $(SRCDIR)/rmc/*pp)
UTILO := $(SRCDIR)/util/util.o

all: example


example: $(SRCDIR)/cppmod/example.o $(SRCDIR)/cppmod/wrapper.cpp
	$(CPP) $(CPPFLAGS) -shared $(PYBIND) $(SRCDIR)/cppmod/wrapper.cpp $(SRCDIR)/cppmod/example.o  -o $(SRCDIR)/cppmod/example$(PY_EXT)

$(SRCDIR)/cppmod/example.o: $(SRCDIR)/cppmod/example.cpp
	$(CPP) $(CPPFLAGS) -c $(SRCDIR)/cppmod/example.cpp -o $(SRCDIR)/cppmod/example.o

rmc: $(RMCSOURCES) util
	$(CPP) $(CPPFLAGS) -c $(SRCDIR)/rmc/RmcCore.cpp -o $(SRCDIR)/rmc/RmcCore.o
	$(CPP) $(CPPFLAGS) -c $(SRCDIR)/rmc/rand.cpp -o $(SRCDIR)/rmc/rand.o
	$(CPP) $(CPPFLAGS) -c $(SRCDIR)/rmc/PlanarRmc.cpp -o $(SRCDIR)/rmc/PlanarRmc.o
	$(CPP) $(CPPFLAGS) -c $(SRCDIR)/rmc/test.cpp -o $(SRCDIR)/rmc/test.o
	$(CPP) $(CPPFLAGS) -shared $(PYBIND) $(SRCDIR)/rmc/wrapper.cpp \
		$(UTILO) $(SRCDIR)/rmc/RmcCore.o $(SRCDIR)/rmc/rand.o \
		$(SRCDIR)/rmc/PlanarRmc.o \
		-o $(SRCDIR)/rmc/rmc$(PY_EXT)
	$(CPP) $(CPPFLAGS) \
		$(UTILO) $(SRCDIR)/rmc/RmcCore.o $(SRCDIR)/rmc/rand.o $(SRCDIR)/rmc/PlanarRmc.o $(SRCDIR)/rmc/test.o \
		-o $(SRCDIR)/rmc/test.out

util: $(SRCDIR)/util/util.cpp
	$(CPP) $(CPPFLAGS) -c $(SRCDIR)/util/util.cpp -o $(SRCDIR)/util/util.o
