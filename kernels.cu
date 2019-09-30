#include "kernels.cuh"

/* ### WORKSET_GEN KERNELS ### */
__global__ void workset_update_BM(char * update, char * bitmap_mask)
{
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    bitmap_mask[tid] = update[tid];

    /* reset update */
    update[tid] = 0;
}

__global__ void workset_update_QU(char * update, queue workset)
{
    int tid = threadIdx.x;
    if (tid == 0)   //first thread clear the workset (no critical section)
    {
        workset.size = 0;
    }
    if (update[tid])
    {
        atomicExch(&workset.items[workset.size], tid);
//        atomicAdd(&workset.size, 1);
    }

    /* reset update */
    update[tid] = 0;
}


/* ### BFS KERNELS ### */
__global__ void one_bfs_B_QU(graph g, queue workset, char * update, int level)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x; 
    if (bid < workset.size) //each block process a workset entry
    {
        int node = workset.items[bid];
        int node_neighbour_index = g.node_vector[node];
        if (tid < (g.node_vector[node+1] - node_neighbour_index))
        {
            //each thread in block process a neighbour of original node of block
            int neighbour = g.edge_vector[node_neighbour_index + tid];
            if (g.node_level_vector[neighbour] > level + 1)
            {
                g.node_level_vector[neighbour] = level + 1;
                update[neighbour] = 1;
            }
        }
    }
}

__global__ void one_bfs_B_BM(graph g, char * bitmap_mask, char * update, int level)
{
    int tid = threadIdx.x;
    int node = blockIdx.x; //each block process a node
    if (bitmap_mask[node])
    {
        int node_neighbour_index = g.node_vector[node];
        if (tid < (g.node_vector[node+1] - node_neighbour_index))
        {
            //each thread in block process a neighbour of original node of block
            int neighbour = g.edge_vector[node_neighbour_index + tid];
            if (g.node_level_vector[neighbour] > level+1)
            {
                g.node_level_vector[neighbour] = level + 1;
                update[neighbour] = 1;
            }
        }
    }
}

__global__ void one_bfs_T_QU(graph g, queue workset, char * update, int level)
{
    int tid = threadIdx.x;
    if (tid < workset.size) //each thread process a workset entry
    {
        int node = workset.items[tid];
        int node_neighbour_index = g.node_vector[node];
        int neighbours_count = g.node_vector[node+1] - node_neighbour_index;
        //visiting neighbours
        for (int neighbour_id = 0; neighbour_id < neighbours_count; neighbour_id++)
        {
            int neighbour = g.edge_vector[node_neighbour_index + neighbour_id];
            if (g.node_level_vector[neighbour] > level+1)
            {
                g.node_level_vector[neighbour] = level + 1;
                update[neighbour] = 1;
            }
        }
    }
}

__global__ void one_bfs_T_BM(graph g, char * bitmap_mask, char * update, int level)
{
    int node = blockDim.x * blockIdx.x + threadIdx.x;
    if (bitmap_mask[node]) //each thread process a node if it's in bitmap_mask
    {
        //visiting neighbours
        int node_neighbour_index = g.node_vector[node];
        int neighbours_count = g.node_vector[node+1] - node_neighbour_index;
        for (int neighbour_id = 0 ; neighbour_id < neighbours_count; neighbour_id++)
        {
            int neighbour = g.edge_vector[node_neighbour_index + neighbour_id];
            if (g.node_level_vector[neighbour] > level)
            {
                g.node_level_vector[neighbour] = level;
                update[neighbour] = 1;
            }
        }
    }
}

/* ### ADD KERNELS ### */
__global__ void add_kernel(char *a_in, int * out) //deprecated
{
	extern __shared__ int a_s[];
	unsigned int tid_block = threadIdx.x;
	unsigned int tid = (blockDim.x*2) * blockIdx.x + tid_block;
    
    a_s[tid_block] = a_in[tid] + a_in[tid+blockDim.x];
	__syncthreads();

    for (unsigned int s = blockDim.x/2; s > 0 ; s >>= 1)
    {
		if (tid_block < s)
			a_s[tid_block] = a_s[tid_block] + a_s[tid_block + s];
		__syncthreads();
	}

    if (tid_block == 0)
        out[blockIdx.x] = a_s[0];
}

__device__ void warpReduce(volatile int* sdata, int tid)
{
	sdata[tid] += sdata[tid + 32];
	sdata[tid] += sdata[tid + 16];
	sdata[tid] += sdata[tid + 8];
	sdata[tid] += sdata[tid + 4];
	sdata[tid] += sdata[tid + 2];
	sdata[tid] += sdata[tid + 1];
}

__global__ void add_kernel_full(char * a_in, int * out)
{
	extern __shared__ int a_s[];
	unsigned int tid_block = threadIdx.x;
	unsigned int tid = (blockDim.x*2) * blockIdx.x + tid_block;
	
	a_s[tid_block] = a_in[tid] + a_in[tid+blockDim.x];
    __syncthreads();

	if (tid_block < 512) {a_s[tid_block] = a_s[tid_block] + a_s[tid_block + 512];} __syncthreads();
	if (tid_block < 256) {a_s[tid_block] = a_s[tid_block] + a_s[tid_block + 256];} __syncthreads();
	if (tid_block < 128) {a_s[tid_block] = a_s[tid_block] + a_s[tid_block + 128];} __syncthreads();
	if (tid_block <  64) {a_s[tid_block] = a_s[tid_block] + a_s[tid_block +  64];} __syncthreads();

	if (tid_block<32) warpReduce(a_s, tid_block);

	if (tid_block == 0){
		out[blockIdx.x] = a_s[0];
	}
}

__global__ void add_kernel_half(char * a_in, int * out)
{
	extern __shared__ int a_s[];
	unsigned int tid_block = threadIdx.x;
	unsigned int tid = (blockDim.x*2) * blockIdx.x + tid_block;
	
	a_s[tid_block] = a_in[tid] + a_in[tid+blockDim.x];
    __syncthreads();

	if (tid_block < 256) {a_s[tid_block] = a_s[tid_block] + a_s[tid_block + 256];} __syncthreads();
	if (tid_block < 128) {a_s[tid_block] = a_s[tid_block] + a_s[tid_block + 128];} __syncthreads();
	if (tid_block <  64) {a_s[tid_block] = a_s[tid_block] + a_s[tid_block +  64];} __syncthreads();

	if (tid_block<32) warpReduce(a_s, tid_block);

	if (tid_block == 0){
		out[blockIdx.x] = a_s[0];
	}
}

/* ### INITAL KERNELS ### */
__global__ void inital_char_array(char * array, char value)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    array[tid] = value;
}

__global__ void inital_int_array(int * array, int value, int size)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < size)
        array[tid] = value;
}

/* ### ALL STEP BFS ### */
__device__ int sum_array_one_thread(int * a_in, int size)
{
    int sum = a_in[0];
    for(int i = 1 ; i < size ; i++)
        sum += a_in[i];
    return sum;
}

#ifdef DP
__global__ void T_BM_bfs(graph g_d, int source, char * bitmap_mask, char * update, argument argument)
{
    int workset_size = 1;
    int level = 0;
    int shared_size = argument.add_block_size * sizeof(int);

    bitmap_mask[source] = 1;
    g_d.node_level_vector[source] = 0;
    
    while(workset_size != 0){
        one_bfs_T_BM<<<argument.covering_block_count, argument.covering_block_size>>>(g_d, bitmap_mask, update, ++level);
        workset_update_BM<<<argument.covering_block_count, argument.covering_block_size>>>(update, bitmap_mask);

        if(argument.add_half_full_flag){
            add_kernel_half<<<argument.add_block_count, argument.add_block_size, shared_size>>>(bitmap_mask, argument.add_result);
        }else{
            add_kernel_full<<<argument.add_block_count, argument.add_block_size, shared_size>>>(bitmap_mask, argument.add_result);
        }

        cudaDeviceSynchronize(); //wait for GPU

        workset_size = sum_array_one_thread(argument.add_result, argument.add_block_count);
    }
}
#endif

/* ### LINIEAR ALGEBRA KERNELS ### */
__global__ void CSR_multiply(graph g_d, int * multiplier_d, int * working_array, int * result_vector, int level)
{
    /* useful names and variables */
    int tid_in_block = threadIdx.x;
    int tid_in_row = threadIdx.x blockIdx.y * blockDim.x;
    int node_vector_index = blockIdx.x;
    
    /* each block save multiplier in its shared memory */
    extern __shared__ int c_s[];
    c_s[tid_in_block] = multiplier_d[tid_in_block];

    /* decode data to find out edge info */
    int edge_vector_offset = g_d.node_vector[node_vector_index];
    int node_degree = g_d.node_vector[node_vector_index+1] - node_start;

    /* sync loading to shared memory */
    __syncthreads();

    /* check if edge exist */
    if(tid_in_row < node_degree){

        /* decode neighbour location and multiply edge wieght-    */
        /* to multiplier element. consider 1 for all edges in BFS */
        int edge_vector_index = edge_vector_offset + tid_in_row;
        int neighbour = g_d.edge_vector[edge_vector_index];
        working_array[edge_vector_index] = c_s[neighbour]; // 1 * c_s[neighbour]
    }

    /* wait for all threads in block to write their result */
    __syncthreads();

    /* reduction add in block: reuse shared memory */
    if (tid_in_block < 512){

        /* check if is there a realated element in result array 512 position away */
        if (tid_in_row+512 < node_degree)
            c_s[tid_in_block] = working_array[tid_in_row] + working_array[tid_in_row + 512]; // blockDim.x/2 = 512
        
        /* check for related element in current position */
        else if(tid_in_row < node_degree)
            c_s[tid_in_block] = working_array[tid_in_row];
        
        /* fill zero for not related elements in shared memory */
        else
            c_s[tid_in_block] = 0; //not neccessery job issue IMPORTANT
    }

    /* wait for first reduction move */
    __syncthreads();

    if (tid_in_block < 256) {c_s[tid_in_block] = c_s[tid_in_block] + c_s[tid_in_block + 256];} __syncthreads();
    if (tid_in_block < 128) {c_s[tid_in_block] = c_s[tid_in_block] + c_s[tid_in_block + 128];} __syncthreads();
    if (tid_in_block <  64) {c_s[tid_in_block] = c_s[tid_in_block] + c_s[tid_in_block +  64];} __syncthreads();
    if (tid_in_block <  32) {warpReduce(c_s, tid_in_block);}

    if (tid_in_block == 0)
        working_array[node_vector_index] = c_s[0];

    __syncthreads();

    /* first thread of each block row */
    if (blockIdx.y == 0 && tid_in_block == 0){
        int result_vector_i = sum_array_one_thread(working_array, gridDim.y); // more potential parallelism

        /* two condition for updating node level vector:                 */
        /* 1- node level must remained untouch                           */
        /* 2- result vector in node position should be anything but zero */
        if(g_d.node_level_vector[node_vector_index] == -1 && result_vector_i != 0)
            g_d.node_level_vector[node_vector_index] = level;

        result_vector[node_vector_index] = result_vector_i;
    }
}