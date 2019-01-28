extern "C"{
#include "structures.h"
#include "desicion_maker.h"
}
#include <limits.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include <unistd.h>
#include "device_launch_parameters.h"
#include <cuda.h>

#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }


#define COVERING_THREAD_PER_BLOCK 1024

/* ### WORKSET_GEN KERNELS ### */
__global__ void workset_update_BM(int * update, int * bitmap_mask)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	bitmap_mask[tid] = update[tid];

	/* reset update */
	update[tid] = 0;
}

__global__ void workset_update_QU(int * update, struct queue * workset) //Deprecated
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid == 0)   //first thread clear the workset (no critical section)
	{
		workset->size = 0;
	}

	__syncthreads();

	if (update[tid])
	{
		//        atomicExch(&workset->items[workset->size], tid);
		//      atomicAdd(&workset->size, 1); //need critical section
	}

	/* reset update */
	update[tid] = 0;
}

__global__ void workset_update_QU_S(int * update, struct queue * workset, int update_size)
{
	//return;
	workset->size = 0;
	for(int i=0 ; i<update_size ; i++)
	{
		if(update[i])
			workset->items[workset->size++] = i;
		update[i] = 0;
	}
}

/* ### BFS KERNELS ### */
__global__ void one_bfs_B_QU(struct graph * g, struct queue * workset, int * update, int level)
{
	int tid = threadIdx.x;  //neighbour count shouldnt extend 1024
	int bid = blockIdx.x; 
	if (bid < workset->size) //each block process a workset entry
	{
		int node = workset->items[bid];
		int node_neighbour_index = g->node_vector[node];
		if (tid < (g->node_vector[node+1] - node_neighbour_index))
		{
			//each thread in block process a neighbour of original node of block
			int neighbour = g->edge_vector[node_neighbour_index + tid];
			if (g->node_level_vector[neighbour] > level + 1)
			{
				g->node_level_vector[neighbour] = level + 1;
				update[neighbour] = 1;
			}
		}
	}
}

__global__ void one_bfs_B_BM(struct graph * g, int * bitmap_mask, int * update, int level)
{
	int tid = threadIdx.x; //neighbour count shouldnt extend 1024
	int node = blockIdx.x; //each block process a node
	if (bitmap_mask[node])
	{
		int node_neighbour_index = g->node_vector[node];
		if (tid < (g->node_vector[node+1] - node_neighbour_index))
		{
			//each thread in block process a neighbour of original node of block
			int neighbour = g->edge_vector[node_neighbour_index + tid];
			if (g->node_level_vector[neighbour] > level+1)
			{
				g->node_level_vector[neighbour] = level + 1;
				update[neighbour] = 1;
			}
		}
	}
}

__global__ void one_bfs_T_QU(struct graph * g, struct queue * workset, int * update, int level)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid < workset->size) //each thread process a workset entry
	{
		int node = workset->items[tid];
		int node_neighbour_index = g->node_vector[node];
		int neighbours_count = g->node_vector[node+1] - node_neighbour_index;
		//visiting neighbours
		for (int neighbour_id = 0; neighbour_id < neighbours_count; neighbour_id++)
		{
			int neighbour = g->edge_vector[node_neighbour_index + neighbour_id];
			if (g->node_level_vector[neighbour] > level+1)
			{
				g->node_level_vector[neighbour] = level + 1;
				update[neighbour] = 1;
			}
		}
	}
}

__global__ void one_bfs_T_BM(struct graph * g, int * bitmap_mask, int * update, int level)
{
	int node = threadIdx.x;
	if (bitmap_mask[node]) //each thread process a node if it's in bitmap_mask
	{
		//visiting neighbours
		int node_neighbour_index = g->node_vector[node];
		int neighbours_count = g->node_vector[node+1] - node_neighbour_index;
		for (int neighbour_id = 0 ; neighbour_id < neighbours_count; neighbour_id++)
		{
			int neighbour = g->edge_vector[node_neighbour_index + neighbour_id];
			if (g->node_level_vector[neighbour] > level+1)
			{
				g->node_level_vector[neighbour] = level + 1;
				update[neighbour] = 1;
			}
		}
	}
}

/* ### DECISION KERNELS ### */
__global__ void add_kernel(int *a_in, int * out)
{
	extern __shared__ int a_s[];
	unsigned int tid_block = threadIdx.x;
	unsigned int tid = (blockDim.x*2) * blockIdx.x + tid_block;
	a_s[tid_block] = a_in[tid] + a_in[tid+blockDim.x];
	__syncthreads();

	for (unsigned int s = blockDim.x/2; s > 0 ; s >>= 1)
	{
		if (tid_block < s){
			a_s[tid_block] = a_s[tid_block] + a_s[tid_block + s];
		}
		__syncthreads();
	}

	if (tid_block == 0){
		out[blockIdx.x] = a_s[0];
	}
}

/* ### INITAL KERNELS ### */
__global__ void inital_char_array(int * array, int value)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	array[tid] = value;
}

__global__ void inital_int_array(struct graph * g_d, int value)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < g_d->size){
		g_d->node_level_vector[tid] = value;
	}
}

/* -------------------------------------------------------------------------------------------- */ 

int sum_array(int *a_in, int size)
{
	int sum = a_in[0];
	for(int i = 1 ; i < size ; i++){
		sum += a_in[i];
	}
	return sum;
}

struct queue * construct_queue_device(int max_size)
{
	// printf("max_size: %d\n", max_size);
	struct queue * workset_h = (struct queue *) malloc(sizeof(struct queue));
	struct queue * workset_d;
	cudaMalloc(&(workset_h->items), sizeof(int)*max_size);

	int zero = 0;
	cudaMemset(&workset_h->items[0], 0, sizeof(int));
	//cudaMemcpy(&workset_h->items[0], &zero, sizeof(int), cudaMemcpyHostToDevice);
	workset_h->size = 1;
	cudaMalloc(&workset_d, sizeof(struct queue));
	cudaMemcpy(workset_d, workset_h, sizeof(struct queue), cudaMemcpyHostToDevice);
	// cudaMemcpy(&workset_d->items[(*workset_size_h)++], &item, sizeof(int), cudaMemcpyHostToDevice);
	// cudaMemcpy(&workset_d->size, workset_size_h, sizeof(int), cudaMemcpyHostToDevice);
	// int * workset_items;
	// cudaMalloc((void **)&workset_items, sizeof(int)*max_size);
	return workset_d;
}

void run_bfs(struct graph * g_h, int source)
{
	/* necessary but not useful variables */
	/* initial workset queue on device */
	struct queue * workset_d = construct_queue_device(g_h->size);
	// int workset_size = 0;
	// queue_push_device(workset_d, source, &workset_size);
	// return;
	/* set and define desicion variables */
	int level = 0, block_count, thread_per_block;
	double avrage_outdeg = get_average_out_deg(g_h);
	int workset_size;
	cudaMemcpy(&workset_size, &workset_d->size, sizeof(int), cudaMemcpyDeviceToHost);
	int algo = decide(avrage_outdeg, workset_size, 1020, &block_count, &thread_per_block); // 
	printf("## INFO: alog: %d\n", algo);
	// return;
	//CUDA_CHECK_RETURN(cudaGetLastError());
	//return;
	int next_sample = next_sample_distance();
	int covering_block_count = (g_h->size - 1)/COVERING_THREAD_PER_BLOCK + 1;
	int update_size = covering_block_count * COVERING_THREAD_PER_BLOCK;
	int * add_result_h = (int *)malloc(sizeof(int)*covering_block_count);
	set_T3(g_h->size);
	/* initial graph on device based on BFS */
	struct graph * g_d;
	
	//cudaMalloc(&g_d, sizeof(struct graph));
	//cudaMemset(&g_d->size,0,sizeof(int));
	//sleep(1);
	//return;
	//CUDA_CHECK_RETURN(cudaGetLastError());
	//return;	
	consturct_graph_device(g_h, &g_d);
	
	int a;
	cudaMemcpy(&a,&g_d->size,sizeof(int),cudaMemcpyDeviceToHost);
	//printf("## INFO: graph constructed\n");
	printf("\n AAAAAA: %d \n", a);
	printf("\n GGGGGG: %d \n", g_h->size);
	//return;
	//sleep(1);
	//return;
	CUDA_CHECK_RETURN(cudaGetLastError());
	//return;
	// return;
	// return;
	//struct graph * g_d;
	//cudaMemcpy(g_d,g_h,sizeof(struct graph),cudaMemcpyHostToDevice);
	//CUDA_CHECK_RETURN(cudaMalloc((void **)&g_d->node_level_vector, sizeof(int)*g_h->size));

	//   cudaMemset(g_d->node_level_vector, INT_MAX, g_h->size);
	//    inital_int_array<<<covering_block_count, COVERING_THREAD_PER_BLOCK>>>(g_d, INT_MAX);
	// return;
	/* initial arrays on device */
	int * update_d;
	int * bitmap_d;
	int * add_result_d;
	// printf("update_size:%d\n")
	// printf("g_h->size:%d\n", g_h->size);
	// printf("update_Size:%d\n", update_size);

	CUDA_CHECK_RETURN(cudaMalloc(&update_d, sizeof(int)*update_size));
	CUDA_CHECK_RETURN(cudaMalloc(&bitmap_d, sizeof(int)*update_size));
	CUDA_CHECK_RETURN(cudaMalloc(&add_result_d, sizeof(int)*covering_block_count));
	printf("## INFO: cudaMalloc done\n");
	//sleep(1);
	//return;
	cudaMemset(update_d, 0, update_size * sizeof(int));
	cudaMemset(bitmap_d, 0, update_size * sizeof(int));
	printf("## INFO: cudaMemset done\n");
	int temp;

	//inital_char_array<<<covering_block_count, COVERING_THREAD_PER_BLOCK>>>(update_d, 0);
	//inital_char_array<<<covering_block_count, COVERING_THREAD_PER_BLOCK>>>(bitmap_d, 0);
	//sleep(3);
	//printf("%d\n", covering_block_count);
	//return;

	/*
	   CUDA_CHECK_RETURN(cudaDeviceSynchronize()); //wait for inital kernels
	   CUDA_CHECK_RETURN(cudaGetLastError());
	 */
	/* bfs first move (workset updated instantly after initialized) */
	cudaMemset(&bitmap_d[source], 1, sizeof(int));
	
	//return;
	
	cudaMemcpy(&temp,&g_d->node_level_vector[0],sizeof(int),cudaMemcpyDeviceToHost);

	printf("asdasdsa : %d \n",temp);
	return;
	cudaMemset(&g_d->node_level_vector[source], 0, sizeof(int));

	sleep(1);
	return;
	printf("## INFO: source init done\n");
	/*
	   CUDA_CHECK_RETURN(cudaMemcpy(&bitmap_d[source], &one, sizeof(int), cudaMemcpyHostToDevice));
	   CUDA_CHECK_RETURN(cudaMemcpy(&g_d->node_level_vector[source], &zero, sizeof(int), cudaMemcpyHostToDevice));
	 */

	cudaMemcpy(&workset_size, &workset_d->size, sizeof(int), cudaMemcpyDeviceToHost);

	//CUDA_CHECK_RETURN(cudaMemcpy(&workset_size, &workset_d->size, sizeof(int), cudaMemcpyHostToDevice));
	printf("workset_d->size: %d\n", workset_size);
	//return;
	//sleep(1);
	//return;
	while (workset_size != 0)
	{	
		printf("algo: %d\n", algo);
		printf("G_QU: %d\n", B_QU);
		if (algo == B_QU)
		{
			printf("1\n");
			while(next_sample--)
			{
				workset_update_QU_S<<<1, 1>>>(update_d, workset_d, update_size);
				one_bfs_B_QU<<<block_count, thread_per_block>>>(g_d, workset_d, update_d, level++);
			}
		} else if (algo == B_BM) 
		{
			printf("2\n");
			while(next_sample--)
			{
				workset_update_BM<<<covering_block_count, COVERING_THREAD_PER_BLOCK>>>(update_d, bitmap_d);
				one_bfs_B_BM<<<covering_block_count, COVERING_THREAD_PER_BLOCK>>>(g_d, bitmap_d, update_d, level++);
			}
		} else if (algo == T_QU)
		{
			printf("3\n");
			while(next_sample--)
			{
				workset_update_QU_S<<<1, 1>>>(update_d, workset_d, update_size);
				one_bfs_T_QU<<<block_count, thread_per_block>>>(g_d, workset_d, update_d, level++);
			}
		} else if (algo == T_BM)
		{
			printf("4\n");
			//return;
			while(next_sample--)
			{
				workset_update_BM<<<covering_block_count, COVERING_THREAD_PER_BLOCK>>>(update_d, bitmap_d);
				one_bfs_T_BM<<<covering_block_count, COVERING_THREAD_PER_BLOCK>>>(g_d, bitmap_d, update_d, level++);
			}
		}
		/* calculate workset size and decide the next move */
		add_kernel<<<covering_block_count, COVERING_THREAD_PER_BLOCK, sizeof(int)*COVERING_THREAD_PER_BLOCK, NULL>>>(update_d, add_result_d);
		printf("end");
		//return;

		CUDA_CHECK_RETURN(cudaDeviceSynchronize()); //wait for GPU
		CUDA_CHECK_RETURN(cudaGetLastError());
		//return;

		CUDA_CHECK_RETURN(cudaMemcpy(add_result_h, add_result_d, sizeof(int)*covering_block_count, cudaMemcpyDeviceToHost));
		workset_size = sum_array(add_result_h, covering_block_count);

		algo = decide(avrage_outdeg, workset_size, 1020, &block_count, &thread_per_block); // mamamd
		next_sample = next_sample_distance();
		//return;
	}

	//return;
	/* return level array of graph to host */
	CUDA_CHECK_RETURN(cudaMemcpy(g_h->node_level_vector, g_d->node_level_vector, sizeof(int)*g_h->size, cudaMemcpyDeviceToHost));

	/* free memory GPU */
	destroy_queue_device(workset_d);
	destroy_graph_device(g_d);
	cudaFree(update_d);
	cudaFree(bitmap_d);
	cudaFree(add_result_d);

	/* free memory CPU */
	free(add_result_h);
}

int main()
{

	struct graph * g = consturct_graph("./dataset/twitter-all.nodes", "./dataset/twitter-all.edges");
	// for (int i = 0; i < g->size && i < 1000; i++) {
	//     printf("%d: %d\n", i, g->node_vector[i]);
	// }
	run_bfs(g, g->node_vector[0]);
	// for(int i=0;)
	// return 0;
}
