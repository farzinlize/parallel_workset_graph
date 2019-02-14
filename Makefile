nvcc_options= -gencode arch=compute_30,code=sm_30 -lm -DLINUX -DDATASET_INDEX=${DATASET} --compiler-options -Wall 
sources = main.cu kernels.cu structures.cpp sequential.cpp desicion_maker.c fuzzy_timing.c

all: ag ag_optimized ag_detailed ag_td

ag: $(sources) Makefile
	nvcc -o build/ag $(sources) $(nvcc_options)

ag_optimized: $(sources) Makefile
	nvcc -o build/ag_optimized $(sources) $(nvcc_options)-O3

ag_detailed: $(sources) Makefile
	nvcc -o build/ag_detailed $(sources) $(nvcc_options)-DCSR_VALIDATION -DDETAIL

ag_td: $(sources) Makefile
	nvcc -o build/ag_td $(sources) $(nvcc_options)-DTEST -DDEBUG

clean:
	rm -rf build/*
