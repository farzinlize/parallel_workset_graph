extern "C"{
    #include "structures.h"
    #include "desicion_maker.h"
    #include "kernels.h"
    #include <limits.h>
}

#define COVERING_THREAD_PER_BLOCK 1024

int sum_array(int *a_in, int size)
{
    int sum = a_in[0];
    for(int i = 1 ; i < size ; i++)
        sum += a_in[i];
    return sum;
}

void run_bfs(struct graph * g_h, int source)
{
    /* necessary but not useful variables */
    int one = 1, zero = 0;

    /* initial workset queue on device */
    struct queue * workset_d = construct_queue_device(g_h->size);
    int workset_size = 0
    queue_push_device(workset_d, source, &workset_size);

    /* set and define desicion variables */
    int level = 0, block_count, thread_per_block;
    double avrage_outdeg = get_average_out_deg(g_h);
    int algo = decide(avrage_outdeg, workset_size, &block_count, &thread_per_block);
    int next_sample = next_sample_distance();
    int covering_block_count = (g_h->size - 1)/COVERING_THREAD_PER_BLOCK + 1;
    int update_size = covering_block_count * COVERING_THREAD_PER_BLOCK;
    int * add_result_h = (int *)malloc(sizeof(int)*covering_block_count);

    /* initial graph on device based on BFS */
    struct graph * g_d = consturct_graph_device(g_h);
    CUDA_CHECK_RETURN(cudaMalloc((void **)&g_d->node_level_vector, sizeof(int)*g_h->size));
    inital_int_array<<<covering_block_count, COVERING_THREAD_PER_BLOCK>>>(g_d->node_level_vector, INT_MAX, g_h->size);
    
    /* initial arrays on device */
    char * update_d, * bitmap_d;
    CUDA_CHECK_RETURN(cudaMalloc((void **)&update_d, sizeof(char)*update_size));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&bitmap_d, sizeof(char)*g_h->size));
    inital_char_array<<<covering_block_count, COVERING_THREAD_PER_BLOCK>>>(update_d, 0);
    inital_char_array<<<covering_block_count, COVERING_THREAD_PER_BLOCK>>>(bitmap_d, 0);

    int * add_result_d;
    CUDA_CHECK_RETURN(cudaMalloc((void **)&add_result_d, sizeof(int)*covering_block_count));

    CUDA_CHECK_RETURN(cudaDeviceSynchronize()); //wait for inital kernels
    CUDA_CHECK_RETURN(cudaGetLastError());

    /* bfs first move (workset updated instantly after initialized) */
    CUDA_CHECK_RETURN(cudaMemcpy(&bitmap_d[source], &one, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(&g_d->node_level_vector[source], &zero, sizeof(int), cudaMemcpyHostToDevice));

    while (workset_size != 0)
    {
        if (algo == B_QU)
        {
            while(next_sample--)
            {
                workset_update_QU<<<covering_block_count, COVERING_THREAD_PER_BLOCK>>>(update_d, workset_d);
                one_bfs_B_QU<<<block_count, thread_per_block>>>(g_d, workset_d, update_d, level++);
            }
        } else if (algo == B_BM) 
        {
            while(next_sample--)
            {
                workset_update_BM<<<covering_block_count, COVERING_THREAD_PER_BLOCK>>>(update_d, bitmap_d);
                one_bfs_B_BM<<<block_count, thread_per_block>>>(g_d, bitmap_d, update_d, level++);
            }
        } else if (algo == T_QU)
        {
            while(next_sample--)
            {
                workset_update_QU<<<covering_block_count, COVERING_THREAD_PER_BLOCK>>>(update_d, workset_d);
                one_bfs_T_QU<<<block_count, thread_per_block>>>(g_d, workset_d, update_d, level++);
            }
        } else if (algo == T_BM)
        {
            while(next_sample--)
            {
                workset_update_BM<<<covering_block_count, COVERING_THREAD_PER_BLOCK>>>(update_d, bitmap_d);
                one_bfs_T_BM<<<block_count, thread_per_block>>>(g_d, bitmap_d, update_d, level++);
            }
        }
        /* calculate workset size and decide the next move */
        add_kernel<<<covering_block_count, COVERING_THREAD_PER_BLOCK, sizeof(int)*COVERING_THREAD_PER_BLOCK>>>(update_d, add_result_d);

        CUDA_CHECK_RETURN(cudaDeviceSynchronize()); //wait for GPU
        CUDA_CHECK_RETURN(cudaGetLastError());
        
        CUDA_CHECK_RETURN(cudaMemcpy(add_result_h, add_result_d, sizeof(int)*covering_block_count, cudaMemcpyDeviceToHost));
        workset_size = sum_array(add_result_h, covering_block_count);

        algo = decide(avrage_outdeg, workset_size, &block_count, &thread_per_block);
        next_sample = next_sample_distance();
    }

    /* return level array of graph to host */
    CUDA_CHECK_RETURN(cudaMemcpy(g_h->node_level_vector, g_d->node_level_vector, sizeof(int)*g->size, cudaMemcpyDeviceToHost));

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
    return 0;
}