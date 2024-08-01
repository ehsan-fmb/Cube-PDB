# Define the compiler to use
CXX = g++
LIBTORCH := ./inc/libtorch


# Define the flags to pass to the compiler
CXXFLAGS = -std=c++17 -Wall -g -pthread -I./inc -I./inc/hog2/graph -I./inc/hog2/envutil -I./inc/hog2/environments \
 -I./inc/hog2/utils -I./inc/hog2/abstraction -I./inc/hog2/simulation -I./inc/hog2/graphalgorithms -I./inc/hog2/generic \
 -I./inc/hog2/algorithms -I./inc/hog2/search -I./inc/hog2/gui -I$(LIBTORCH)/include -I$(LIBTORCH)/include/torch/csrc/api/include \
 -I/usr/local/cuda-12.4/include

# Library paths (where to find the libraries)
CUDA_FLAGS := -L/usr/local/cuda-12.4/lib64 -lcudart
LDFLAGS = -Wl,-no-undefined -Wl,--no-as-needed \
-L./inc/hog2/bin/release -L$(LIBTORCH)/lib -ltorch -ltorch_cpu -ltorch_cuda -lc10_cuda -lc10 -Wl,-rpath,$(LIBTORCH)/lib $(CUDA_FLAGS)
LDLIBS = -lgraph -lenvironments -lenvutil -lmapalgorithms -lalgorithms -lgraphalgorithms -lutils  -lSTUB 

# Define the directories
SRC_DIR = src
OUTPUT=bin/main
OBS_DIR = objs
   
# List source files
SRC = $(SRC_DIR)/main.cpp

# Object files
OBJ = $(OBS_DIR)/main.o

# Default target
TARGET = $(OUTPUT)
all: $(TARGET)

# Rule to link in one step
$(OUTPUT): $(OBJ) 
	$(CXX) $(CXXFLAGS) -o $@ $(OBJ) $(LDFLAGS) $(LDLIBS) 

# Rules to compile source files into object files
$(OBJ): $(SRC)
	$(CXX)	-c	$(CXXFLAGS)	-o	$@	$<

# Clean up generated files
clean:
	rm -f $(OBJ) $(OUTPUT)

.PHONY: all clean