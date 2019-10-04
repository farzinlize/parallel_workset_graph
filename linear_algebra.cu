#include "linear_algebra.cuh"

__device__ int sum_1d2d(int * a, int start, int end)
{
    int sum = a[start];
    for(int i=start+1;i<end;i++){
        sum += a[i];
    }
    return sum;
}

__device__ void warpReducePlus(volatile int* sdata, int tid)
{
	sdata[tid] += sdata[tid + 32];
	sdata[tid] += sdata[tid + 16];
	sdata[tid] += sdata[tid + 8];
	sdata[tid] += sdata[tid + 4];
	sdata[tid] += sdata[tid + 2];
	sdata[tid] += sdata[tid + 1];
}

/* ### LINIEAR ALGEBRA KERNELS ### */
#ifdef DP
__global__ void linear_algebra_bfs_dp(graph g_d, int source, int * multiplier_d, int * working_array, argument_la argument)
{
    int level = 0;
    multiplier_d[source] = 1;

    int i=0;
    for(i=0 ; i < argument.max_depth ; i++){
        printf("shared size = %d\n", argument.shared_size);
        CSR_multiply_one_BFS<<<argument.grid_dim, argument.block_dim, argument.shared_size>>>(g_d, multiplier_d, working_array, ++level);
        cudaDeviceSynchronize(); //wait for result
    }
}
#endif

__global__ void CSR_multiply_reductionAdd(graph g_d, int * multiplier_d, int * working_array)
{
    /* useful names and variables */
    int tid_in_block = threadIdx.x;
    int tid_in_row = threadIdx.x + blockIdx.y * blockDim.x;
    int node_vector_index = blockIdx.x;
    
    /* define shared memory */
    extern __shared__ int c_s[];

    /* decode data to find out edge info */
    int edge_vector_offset = g_d.node_vector[node_vector_index];
    int node_degree = g_d.node_vector[node_vector_index+1] - edge_vector_offset;

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
    if (tid_in_block <  32) {warpReducePlus(c_s, tid_in_block);}

    if (tid_in_block == 0)
        working_array[node_vector_index * gridDim.y + blockIdx.y] = c_s[0];
}

__global__ void result_and_BFS(graph g_d, int * multiplier_d, int * working_array, int block_count_y, int level)
{
    /* each thread are resposible of one result index */
    int node_vector_index = threadIdx.x + blockDim.x * blockIdx.x;

    if(node_vector_index < g_d.size){
        int result_vector_i = sum_1d2d(working_array, node_vector_index * block_count_y, (node_vector_index+1) * block_count_y); // more potential parallelism

        /* two condition for updating node level vector:                 */
        /* 1- node level must remained untouch                           */
        /* 2- result vector in node position should be anything but zero */
        if(result_vector_i != 0 && g_d.node_level_vector[node_vector_index] == -1)
            g_d.node_level_vector[node_vector_index] = level;
    
        /* rewrite on multiplier_d for next round                     */
        /* ignore exact result by replacing 1 for any non-zero result */
        if(result_vector_i != 0)
            multiplier_d[node_vector_index] = 1;
        else
            multiplier_d[node_vector_index] = 0;
    }
}

/* ### MAIN ### */
void linear_algebra_bfs(graph g_h, int source)
{
    /* define useful variables */
    int maximum_threads_in_block = 1024;
    int one = 1, zero = 0, max_depth = 10;

    /* initial graph on device based on BFS */
    graph g_d = consturct_graph_device(g_h);
    CUDA_CHECK_RETURN(cudaMalloc((void **)&(g_d.node_level_vector), sizeof(int)*g_h.size));
    CUDA_CHECK_RETURN(cudaMemset(g_d.node_level_vector, -1, sizeof(int)*g_h.size));

    #ifndef DP
    /* set source node level to zero */
    CUDA_CHECK_RETURN(cudaMemcpy(&g_d.node_level_vector[source], &zero, sizeof(int), cudaMemcpyHostToDevice));
    #endif

    /* kernel grids configuration variables */
    int covering_block_count = (g_h.size - 1)/maximum_threads_in_block + 1;
    dim3 grid_dim_mult(g_h.size, covering_block_count, 1);
    dim3 block_dim_mult(maximum_threads_in_block, 1, 1);
    dim3 grid_dim_result(covering_block_count, 1, 1);
    dim3 block_dim_result(maximum_threads_in_block, 1, 1);

    /* initial GPU arrays */
    int * multiplier_d;
    int * working_array;
    CUDA_CHECK_RETURN(cudaMalloc((void **)&(multiplier_d), sizeof(int)*g_h.size));
    CUDA_CHECK_RETURN(cudaMemset(multiplier_d, 0, sizeof(int)*g_h.size));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&(working_array), sizeof(int) * g_h.size * covering_block_count));

    #ifndef DP
    /* set first multiplier vector */
    CUDA_CHECK_RETURN(cudaMemcpy(&multiplier_d[source], &one, sizeof(int), cudaMemcpyHostToDevice));
    #endif

    #ifdef DETAIL
    int * multiplier_h = (int *)malloc(sizeof(int) * g_h.size);
    #endif

    #ifdef DP
    /* set arguments */
    argument_la argument_d;
    argument_d.grid_dim = grid_dim;
    argument_d.block_dim = block_dim;
    argument_d.shared_size = maximum_threads_in_block * sizeof(int);
    argument_d.max_depth = max_depth;
    #endif

    /* call DP kernel */
    #ifndef DP
    int level = 0;

        /* test in non-DP mode */
        #ifdef DETAIL
        FILE * mult_report;
        mult_report = fopen("out/mult_report.out", "wb");
        int j;
        #endif

    int i=0;
    for(i=0 ; i < max_depth ; i++){
        CSR_multiply_reductionAdd<<<grid_dim_mult, block_dim_mult, (maximum_threads_in_block*sizeof(int))>>>(g_d, multiplier_d, working_array);
        result_and_BFS<<<grid_dim_result, block_dim_result>>>(g_d, multiplier_d, working_array, covering_block_count, ++level);

        printf("multiply number operating: %d |\t", i);
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());
        printf("done\n");

        #ifdef DETAIL
        /* check for multiply result each step for test */
        CUDA_CHECK_RETURN(cudaMemcpy(multiplier_h, multiplier_d, sizeof(int)*g_h.size, cudaMemcpyDeviceToHost));

        fprintf(mult_report, "index = %d\n[", i);
        for(j=0;j<g_h.size;j++){
            fprintf(mult_report, "%d\t", multiplier_h[j]);
        }
        fprintf(mult_report, "]\n");
        #endif
    }
    #else
    linear_algebra_bfs_dp<<<1, 1>>>(g_d, source, multiplier_d, working_array, argument_d);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    #endif

    /* return level array of graph to host */
    CUDA_CHECK_RETURN(cudaMemcpy(g_h.node_level_vector, g_d.node_level_vector, sizeof(int)*g_h.size, cudaMemcpyDeviceToHost));

    /* free memory */
    destroy_graph_device(g_d);
    cudaFree(g_d.node_level_vector);
    cudaFree(multiplier_d);
    #ifdef DETAIL
    free(multiplier_h);
    fclose(mult_report);
    #endif
    cudaFree(working_array);
}