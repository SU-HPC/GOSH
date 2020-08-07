DIRS= execs# directories to be created

# intermediate files
OBJECTS = sigmoid.o rand_helper.o gpu_functions.o big_graphs.o
ARGPARSE_MAIN = main_argparse.o
NONARGPARSE_MAIN = main.o 
TARGET_ARGPARSE=execs/gosh.out
TARGET_NONARGPARSE=execs/gosh_nargparse.out
MKDIR_C=mkdir -p

# compilers and flags
CXX=g++ #compiler
NVCC=nvcc
NVCCFLAGS= -std=c++11 -Xcompiler -fopenmp -lgomp -O3  #-lnvToolsExt  -g -G #flags 

all: $(TARGET_ARGPARSE) $(TARGET_NONARGPARSE) .intr

$(TARGET_ARGPARSE) : dir $(OBJECTS) $(ARGPARSE_MAIN)
	$(NVCC) $(NVCCFLAGS) $(OBJECTS) $(ARGPARSE_MAIN) -o $@ 

$(TARGET_NONARGPARSE) : dir $(OBJECTS) $(NONARGPARSE_MAIN)
	$(NVCC) $(NVCCFLAGS) $(OBJECTS) $(NONARGPARSE_MAIN) -o $@ 

%.o: src/utils/%.cpp
	$(NVCC) -x cu $(NVCCFLAGS) -I. -dc $< -o $@

%.o: src/utils/%.cu
	$(NVCC) $(NVCCFLAGS) -D_BIG_GRAPHS  -I. -dc $< -o $@

%.o: src/%.cu
	$(NVCC) $(NVCCFLAGS) -I. -dc $< -o $@

$(ARGPARSE_MAIN) : src/main.cu
	$(NVCC) $(NVCCFLAGS) -D_ARGPARSE -I. -dc $< -o $@

$(NONARGPARSE_MAIN) : src/main.cu
	$(NVCC) $(NVCCFLAGS) -I. -dc $< -o $@

dir:
	$(MKDIR_C) $(DIRS)

.intr:
	rm -rf $(OBJECTS) $(ARGPARSE_MAIN) $(NONARGPARSE_MAIN)

.PHONY: clean

clean:
	@rm -rfv  $(OBJECTS) $(TARGET_ARGPARSE) $(TARGET_NONARGPARSE)
