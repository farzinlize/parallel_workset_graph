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
        #ifndef EXACT_MULT
        if(result_vector_i != 0)
            multiplier_d[node_vector_index] = 1;
        else
            multiplier_d[node_vector_index] = 0;
        #else
        multiplier_d[node_vector_index] = result_vector_i;
        #endif
    }
}

__global__ void spmv_csr_scalar_kernel (graph g_d, int * multiplier_d, int * working_array, int level)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if(row < g_d.size){
        int dot = 0;
        int row_start = g_d.node_vector[row];
        int row_end = g_d.node_vector[row + 1];
        for (int edge_vector_index = row_start ; edge_vector_index < row_end ; edge_vector_index ++)
            dot +=  multiplier_d[g_d.edge_vector[edge_vector_index]];

        if(dot != 0 && g_d.node_level_vector[row] == -1)
            g_d.node_level_vector[row] = level;

        #ifdef EXACT_MULT
        working_array[row] = dot;
        #else
        if(dot != 0)
            working_array[row] = 1;
        else
            working_array[row] = 0;
        #endif
    }
}

__global__ void set_result (graph g_d, int * multiplier_d, int * working_array)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if(row < g_d.size){
        multiplier_d[row] = working_array[row];
    }
}
