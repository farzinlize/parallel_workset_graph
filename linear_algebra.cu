#include "linear_algebra.cuh"

/* ### LINIEAR ALGEBRA KERNELS ### */
__global__ void linear_algebra_bfs(graph g_d, int source, int * multiplier_d, int ** working_array, argument_la argument)
{
    int level = 0;
    multiplier_d[source] = 1;

    int i=0;
    for(i=0 ; i < argument.max_depth ; i++){
        CSR_multiply_one_BFS<<<argument.grid_dim, argument.block_dim, argument.shared_size>>>(g_d, multiplier_d, working_array, ++level);
        cudaDeviceSynchronize(); //wait for result
    }
}

__global__ void CSR_multiply_one_BFS(graph g_d, int * multiplier_d, int ** working_array, int level)
{
    /* useful names and variables */
    int tid_in_block = threadIdx.x;
    int tid_in_row = threadIdx.x + blockIdx.y * blockDim.x;
    int node_vector_index = blockIdx.x;
    
    /* each block save multiplier in its shared memory */
    extern __shared__ int c_s[];

    /* decode data to find out edge info */
    int edge_vector_offset = g_d.node_vector[node_vector_index];
    int node_degree = g_d.node_vector[node_vector_index+1] - edge_vector_offset;

    /* sync loading to shared memory */
    __syncthreads();

    /* check if edge exist */
    if(tid_in_row < node_degree){

        /* decode neighbour location and multiply edge wieght-    */
        /* to multiplier element. consider 1 for all edges in BFS */
        int edge_vector_index = edge_vector_offset + tid_in_row;
        int neighbour = g_d.edge_vector[edge_vector_index];

        /* save multiplication result in shared memory */
        c_s[tid_in_block] = multiplier_d[neighbour]; // 1 * nultiplier_d[neighbour]
    }

    /* wait for all threads in block to write their result */
    __syncthreads();

    /* reduction add in blocks */
    if (tid_in_block < 512){

        /* check if is there a realated element in shared memory 512 position away */
        if (tid_in_row+512 < node_degree){
            c_s[tid_in_block] = c_s[tid_in_block] + c_s[tid_in_block + 512]; // blockDim.x/2 = 512
        }
        /* check for related data at current position (if isn't there, load 0 in shared memory) */
        else if(tid_in_row >= node_degree){
            c_s[tid_in_block] = 0; //not neccessery job issue IMPORTANT
        }
    }

    /* wait for first reduction move */
    __syncthreads();

    if (tid_in_block < 256) {c_s[tid_in_block] = c_s[tid_in_block] + c_s[tid_in_block + 256];} __syncthreads();
    if (tid_in_block < 128) {c_s[tid_in_block] = c_s[tid_in_block] + c_s[tid_in_block + 128];} __syncthreads();
    if (tid_in_block <  64) {c_s[tid_in_block] = c_s[tid_in_block] + c_s[tid_in_block +  64];} __syncthreads();
    if (tid_in_block <  32) {warpReduce(c_s, tid_in_block);}

    if (tid_in_block == 0)
        working_array[node_vector_index][blockIdx.y] = c_s[0];

    __syncthreads();

    /* first thread of each block row */
    if (blockIdx.y == 0 && tid_in_block == 0){
        int result_vector_i = sum_array_one_thread(working_array[node_vector_index], gridDim.y); // more potential parallelism

        /* two condition for updating node level vector:                 */
        /* 1- node level must remained untouch                           */
        /* 2- result vector in node position should be anything but zero */
        if(g_d.node_level_vector[node_vector_index] == -1 && result_vector_i != 0)
            g_d.node_level_vector[node_vector_index] = level;

        multiplier_d[node_vector_index] = result_vector_i; // rewrite on multiplier_d for next round
    }
}

/* ### MAIN ### */
void linear_algebra_bfs(graph g_h, int source)
{
    int maximum_threads_in_block = 1024;

    /* initial graph on device based on BFS */
    graph g_d = consturct_graph_device(g_h);
    CUDA_CHECK_RETURN(cudaMalloc((void **)&(g_d.node_level_vector), sizeof(int)*g_h.size));
    CUDA_CHECK_RETURN(cudaMemset(g_d.node_level_vector, -1, sizeof(int)*g_h.size)); // dose it works?

    /*  */
    int covering_block_count = (g_h.size - 1)/maximum_threads_in_block + 1;
    dim3 grid_dim(g_h.size, covering_block_count, 1);
    dim3 block_dim(maximum_threads_in_block, 1, 1);

    /* initial GPU arrays */
    int * multiplier_d;
    CUDA_CHECK_RETURN(cudaMalloc((void **)&(multiplier_d), sizeof(int)*g_h.size));

    /* 2D GPU array initialization */
    int ** someHostArray = (int **) malloc(sizeof(int *) * g_h.size);
    int ** working_array;
    CUDA_CHECK_RETURN(cudaMalloc((void **)&(working_array), sizeof(int *) * g_h.size));
    for(int i=0;i<g_h.size;i++)
        CUDA_CHECK_RETURN(cudaMalloc((void **)&(someHostArray[i]), sizeof(int) * covering_block_count));
    CUDA_CHECK_RETURN(cudaMemcpy(working_array, someHostArray, g_h.size * sizeof(int *), cudaMemcpyHostToDevice));

    /* set arguments */
    argument_la argument_d;
    argument_d.grid_dim = grid_dim;
    argument_d.block_dim = block_dim;
    argument_d.shared_size = maximum_threads_in_block * sizeof(int);
    argument_d.max_depth = 10;

    /* call DP kernel */
    linear_algebra_bfs<<<1, 1>>>(g_d, source, multiplier_d, working_array, argument_d);

    /* free memory */
    destroy_graph_device(g_d);
    cudaFree(g_d.node_level_vector);
    cudaFree(multiplier_d);
    for(int i=0;i<g_h.size;i++)
        cudaFree(someHostArray[i]);
    cudaFree(working_array);
    free(someHostArray);
}