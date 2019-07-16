# Parallel Graph Algorithms
Deploying graph algorithms on GPUs with an adaptive solution using work-set for parallelization. In each step program decide to use a solution based on work-set size and other elements in the graph to gain speedup from preventing not efficient solution as much as parallelization. Michela Becchi and Da Li Paper of *"Deploying Graph Algorithms on GPUs: an Adaptive Solution"* consider the base for this research-based project. 

## Work Set Types
As described in related paper this project uses two kind of work-sets. A bitmap covering all threads that shows which thread has job and which one doesn't. And a work-set containing node IDs that need processing. Any thread with a thread ID less than work-set size process one element of work-set in parallel. In both type, work-set size determine parallelization level because in each step threads process every elements of work-set in parallel. Kernels or any functions working with bitmaps has *BM* in its name and others working with work-set uses *QU* in its name as *queue*.

### Queue work-set
unrelated to its name, this data structure differs from known queues. Elements will added sequentially but any element are accessible in any time because the work-set will be empty at the end of each step in our program at once.

## Approach: CPU and GPU Part
The approach is to use CPU in charge of deciding solution (adaptive part) and launching related kernels and GPU in each step processing all work-set and updating it using another kernel. Main application follows these steps in summary:

1.	Initialing arrays and data structures on both host and device
2.	Apply first move *(for example: adding source to work-set in BFS algorithm)*
3.	Launch decided algorithm kernel with work-set updater (described in GPU kernels section)
4.	Collect new data to make new decision at each *k* algorithm steps
5.	Do step 3 and 4 until work-set get empty

## GPU Kernels
All kernels implemented in the same file named `kernels.cu` with `kernels.cuh` as header. There is one kernel corresponding each solution that finish one step of processing the algorithm. These kernels differs in two type of categories. First category involve with thread or thread blocks.
If in a kernel each thread process a node it categorize as thread based kernel with *T* in its name and if each thread block process a node it categorize as block based kernel with *B* in its name. In a block based kernel each thread in block presents neighbors of the original node but in thread base kernel each thread visit each of its neighbors by itself.
Different work-set type also need separated kernels (for each block and thread base) 
Also there is two kernel (for two type of work-set) for updating next work-set. All kernels that consume work-set elements, produce a bitmap array covering all nodes in graph called `update` that shows which nodes should be in the next work-set.

 

> This Report Written by Farzin Mohammdi with [StackEdit](https://stackedit.io/).
