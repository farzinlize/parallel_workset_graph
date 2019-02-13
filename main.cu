#include "kernels.cuh"
#include <stdio.h>
#include <limits.h>
#include "structures.h"
#include "sequential.h"

extern "C"{
    #include "desicion_maker.h"
    #include "fuzzy_timing.h"
}

#define COVERING_THREAD_PER_BLOCK 1024
#define DATASET_COUNT 1

const char * dataset_files[DATASET_COUNT][2] = {{"dataset/twitter-all.nodes", "dataset/twitter-all.edges"}};

int sum_array(int *a_in, int size)
{
    int sum = a_in[0];
    for(int i = 1 ; i < size ; i++)
        sum += a_in[i];
    return sum;
}

void run_bfs(graph g_h, int source)
{
    /* necessary but not useful variables */
    int one = 1, zero = 0;

    /* initial workset queue on device (instantly add first bfs move) */
    int workset_size_h = 1;
    queue workset_d = construct_queue_device_with_source(g_h.size, &source);

    /* set and define desicion variables */
    int level = 0;
    double avrage_outdeg = get_average_out_deg(g_h);
    int algo = decide(avrage_outdeg, workset_size_h);
    int next_sample = next_sample_distance();
    int covering_block_count = (g_h.size - 1)/COVERING_THREAD_PER_BLOCK + 1;
    int update_size = covering_block_count * COVERING_THREAD_PER_BLOCK;
    int * add_result_h = (int *)malloc(sizeof(int)*covering_block_count);

    /* initial graph on device based on BFS */
    graph g_d = consturct_graph_device(g_h);
    CUDA_CHECK_RETURN(cudaMalloc((void **)&(g_d.node_level_vector), sizeof(int)*g_h.size));
    CUDA_CHECK_RETURN(cudaMemset(g_d.node_level_vector, 0, sizeof(int)*g_h.size)); //WRONG! INT_MAX value
    
    /* initial arrays on device */
    char * update_d, * bitmap_d;
    CUDA_CHECK_RETURN(cudaMalloc((void **)&update_d, sizeof(char)*update_size));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&bitmap_d, sizeof(char)*update_size));
    CUDA_CHECK_RETURN(cudaMemset(update_d, 0, sizeof(char)*update_size));
    CUDA_CHECK_RETURN(cudaMemset(bitmap_d, 0, sizeof(char)*update_size));

    int * add_result_d;
    CUDA_CHECK_RETURN(cudaMalloc((void **)&add_result_d, sizeof(int)*covering_block_count));

    /* bfs first move (workset updated instantly after initialized) */
    CUDA_CHECK_RETURN(cudaMemcpy(&bitmap_d[source], &one, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(&g_d.node_level_vector[source], &zero, sizeof(int), cudaMemcpyHostToDevice));

    while (workset_size_h != 0)
    {
        if (algo == B_QU)
        {
            while(next_sample--)
            {
                workset_update_QU<<<covering_block_count, COVERING_THREAD_PER_BLOCK>>>(update_d, workset_d);
                one_bfs_B_QU<<<covering_block_count, COVERING_THREAD_PER_BLOCK>>>(g_d, workset_d, update_d, level++);
            }
        } else if (algo == B_BM) 
        {
            while(next_sample--)
            {
                workset_update_BM<<<covering_block_count, COVERING_THREAD_PER_BLOCK>>>(update_d, bitmap_d);
                one_bfs_B_BM<<<covering_block_count, COVERING_THREAD_PER_BLOCK>>>(g_d, bitmap_d, update_d, level++);
            }
        } else if (algo == T_QU)
        {
            while(next_sample--)
            {
                workset_update_QU<<<covering_block_count, COVERING_THREAD_PER_BLOCK>>>(update_d, workset_d);
                one_bfs_T_QU<<<covering_block_count, COVERING_THREAD_PER_BLOCK>>>(g_d, workset_d, update_d, level++);
            }
        } else if (algo == T_BM)
        {
            while(next_sample--)
            {
                workset_update_BM<<<covering_block_count, COVERING_THREAD_PER_BLOCK>>>(update_d, bitmap_d);
                one_bfs_T_BM<<<covering_block_count, COVERING_THREAD_PER_BLOCK>>>(g_d, bitmap_d, update_d, level++);
            }
        }
        /* calculate workset size and decide the next move */
        add_kernel<<<covering_block_count, COVERING_THREAD_PER_BLOCK, sizeof(int)*COVERING_THREAD_PER_BLOCK>>>(update_d, add_result_d);

        CUDA_CHECK_RETURN(cudaDeviceSynchronize()); //wait for GPU
        CUDA_CHECK_RETURN(cudaGetLastError());
        
        CUDA_CHECK_RETURN(cudaMemcpy(add_result_h, add_result_d, sizeof(int)*covering_block_count, cudaMemcpyDeviceToHost));
        workset_size_h = sum_array(add_result_h, covering_block_count);

        algo = decide(avrage_outdeg, workset_size_h);
        next_sample = next_sample_distance();
    }

    /* return level array of graph to host */
    CUDA_CHECK_RETURN(cudaMemcpy(g_h.node_level_vector, g_d.node_level_vector, sizeof(int)*g_h.size, cudaMemcpyDeviceToHost));

    /* free memory GPU */
    destroy_queue_device(workset_d);
    destroy_graph_device(g_d);
    cudaFree(g_d.node_level_vector);
    cudaFree(update_d);
    cudaFree(bitmap_d);
    cudaFree(add_result_d);

    /* free memory CPU */
    free(add_result_h);
}

#ifndef TEST
int main(int argc, char * argv[])
{
    printf("[MAIN] app.cu main\tDataset index: %d\n", DATASET_INDEX);

    /* read data set */
    graph g_h = consturct_graph(dataset_files[DATASET_INDEX][0], dataset_files[DATASET_INDEX][1]);

    #ifdef DEBUG
    printf("[DEBUG][MAIN] running sequential bfs with graph size: %d\n", g_h.size);
    #endif

    set_clock();

    sequential_run_bfs_QU(&g_h, 0)

    double elapced = get_elapsed_time();

    #ifdef DEBUG
    printf("[DEBUG][MAIN] returning sequential bfs, time: %.2f\n", elapced);
    #endif

    free(g_h.node_level_vector);
    destroy_graph(g_h);

    return 0;
}
#else

__global__ void testKernel(queue * q)
{
    printf("queue size at the beginning of kernel: %d\n", q->size);

    for(int i=0;i<q->size;i++)
        printf("items id: %d\titem value: %d\n", i, q->items[i]);

    q->size = 2;

    printf("queue size at the end of kernel: %d\n", q->size);
}

int main(int argc, char * argv[])
{
    printf("[MAIN] test main at app.cu\tDataset index: %d\n", DATASET_INDEX);

    queue * test;
    printf("1\n");
    cudaMallocManaged(&test, sizeof(queue));
    cudaMallocManaged(&test->items, sizeof(int)*20);
    printf("2\n");
    test->items[0] = 85;
    printf("2.5\n");
    test->items[1] = 95;
    test->items[2] = 29;
    test->items[3] = 55;
    test->items[4] = 33;

    printf("3\n");
    test->size = 5;

    printf("4\n");
    testKernel<<<1, 1>>>(test);

    cudaDeviceSynchronize();
    
    printf("queue size after kernel in main: %d\n", test->size);

    cudaFree(test->items);

    return 0;
}
#endif