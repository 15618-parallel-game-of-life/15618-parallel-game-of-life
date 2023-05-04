CU_FILES   := cudaRenderer.cu
CC_FILES   := main.cpp display.cpp
EXECUTABLE := render
OBJDIR = objs
OBJS = $(OBJDIR)/main.o $(OBJDIR)/display.o $(OBJDIR)/cudaRenderer.o
LOGS = outputs/*.log

ARCH = $(shell uname | sed -e 's/-.*//g')
CXX = g++ -m64
CXXFLAGS = -O3 -Wall -fopenmp
LDFLAGS = -L/usr/local/cuda-11.7/lib64/ -lcudart
NVCC = nvcc
NVCCFLAGS = -O3 -m64 --gpu-architecture compute_61 -ccbin /usr/bin/gcc
HOSTNAME = $(shell hostname)

LIBS := GL glut cudart
LDLIBS := $(addprefix -l, $(LIBS))

all: $(EXECUTABLE)

default: $(EXECUTABLE)

dirs:
		@mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *~ $(EXECUTABLE) $(LOGS)

$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS) $(LDLIBS) $(LDFRAMEWORKS)

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: %.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@

.PHONY: dirs clean
