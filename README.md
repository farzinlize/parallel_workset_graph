
# Parallel Graph Algorithms

Deploying graph algorithms on GPUs with an adaptive solution using work-set for parallelization. In each step program decide to use a solution based on work-set size and other elements in the graph to gain speedup from preventing not efficient solution as much as parallelization. Michela Becchi and Da Li Paper of *"Deploying Graph Algorithms on GPUs: an Adaptive Solution"* consider the base for this research-based project.

### Update: Linear Algebra Approach

There are also separated modules in this project for implementing graph algorithms using linear algebra. `linear_algebra.cu` with `linear_algebra.cuh` as its header contains kernels that implementing linear algebra logic using matrix product with CSR format.


# Adaptive Solution

In this part I describe some aspect of Michela Becchi paper: *"Deploying Graph Algorithms on GPUs: an Adaptive Solution"* and how I tried to implement it.

## GPU Kernels

All kernels related to *Michela Becchi paper* implemented in the same file named `kernels.cu` with `kernels.cuh` as header. There is one kernel corresponding each solution that finish one step of processing the algorithm. These kernels differs in two type of categories. First category involve with thread or thread blocks.

If in a kernel each thread process a node it categorize as thread based kernel with *T* in its name and if each thread block process a node it categorize as block based kernel with *B* in its name. In a block based kernel each thread in block presents neighbors of the original node but in thread base kernel each thread visit each of its neighbors by itself.

Different work-set type also need separated kernels (for each block and thread base)
 
Also there is two kernel (for two type of work-set) for updating next work-set. All kernels that consume work-set elements, produce a bitmap array covering all nodes in graph called `update` that shows which nodes should be in the next work-set.


## Work Set Types

As described in related paper this project uses two kind of work-sets. A bitmap covering all threads that shows which thread has job and which one doesn't. And a work-set containing node IDs that need processing. Any thread with a thread ID less than work-set size process one element of work-set in parallel. In both type, work-set size determine parallelization level because in each step threads process every elements of work-set in parallel. Kernels or any functions working with bitmaps has *BM* in its name and others working with work-set uses *QU* in its name as *queue*.


### Queue work-set

unrelated to its name, this data structure differs from known queues. Elements will added sequentially but any element are accessible in any time because the work-set will be empty at the end of each step in our program at once.

  

## Approach: CPU and GPU Part

The approach is to use CPU in charge of deciding solution (adaptive part) and launching related kernels and GPU in each step processing all work-set and updating it using another kernel. Main application of this part follows these steps in summary:

  

1. 	Initialing arrays and data structures on both host and device

2. 	Apply first move *(for example: adding source to work-set in BFS algorithm)*

3. 	Launch decided algorithm kernel with work-set updater (described in GPU kernels section)

4. 	Collect new data to make new decision at each *k* algorithm steps

5. 	Do step 3 and 4 until work-set get empty

 

# Project Structure

This Project uses separated modules to achieves its goal and here is the list of all modules with short description:

 - `main.cu` : main application file, responsible for using other modules and make reports
 - `kernels.cu` : contains all kernels of *Michela Becchi paper* and other needed one
 - `linear_algebra.cu` : contains linear algebra kernels
 - `linear_algebra_main.cu` : contains main function responsible of calling linear algebra kernels
 - `sequential.cpp` : contains all sequential implementations
 - `structures.cpp` : contains functions responsible of reading and processing graph structures if form of CSR
 - `desicion_maker.c` : decision making functions related to *Michela Becchi paper*
 - `fuzzy_timing.c` : timing modules (all function in this module except `gettimeofday` function provided by [_Amir Hossein Sojoodi_](https://github.com/amirsojoodi) at repository of [GPGPU-2018Fall](https://github.com/amirsojoodi/GPGPU-2018Fall) hosted on github. 
 - `report.cpp` : this module is responsible of initialing report file and export different reports for validation and debug
 - `make.bat` : a batch file for compilation in windows
 - `Makefile` : make file for Linux based compilation

  

## Application Report
All information printed middle of the application will stored in a file that named using application version and compile modes. Version specify in `make.bat` file and any upgrade also specify in a commit. There is a separated module, named `report.h`  that responsible for all other reports except middle application prints. 

### Compare BFS result file
One of the report types in this project is a comparison report between two BFS implementation run. A function named `make_compare_file` that receive a file name and two level vector with their implementation names, will write a report file including these parts:

 - Each node id with it's level in two given level
 - Max level in both implementation
 - Number of nodes that belong to the tree in one implementation but not in the another one (FAULT TREE)
 - Number of nodes with different level in two implementation
 - Number of nodes in each level

## Compile mode
This project has a batch file for compile specific modes of application. These types are related to reported data and using different type of implementation as described at below table. Also there is a Makefile for Linux compilation and it contains several rules to make different type of executable files.

| Flag | Description |
|--|--|
| csr | Program checks for standard CSR data-set and run relevant tests |
| dp | Run program in dynamic parallelism mode (described below) |
| detail | Program provides more information as report |
| debug | Activate debug part of codes |
| test | Compile a different main function for testing purposes |

Also there are some options that need one following argument that specify with a dash in their name. These options are described at below table:

| Option | Description |
|--|--|
| dataset | Uses an integer as argument that indicates index of desired data-set |
| name | Changes name of executable file with its followed argument |



 

  

> This Report Written by Farzin Mohammdi with [StackEdit](https://stackedit.io/).