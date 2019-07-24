#include "kernels.cuh"
#include <stdio.h>
#include <limits.h>
#include "structures.h"
#include "sequential.h"
#include "report.h"
#include <limits.h>

extern "C"{
    #include "desicion_maker.h"
    #include "fuzzy_timing.h"
}

#define COVERING_THREAD_PER_BLOCK 1024
#define DATASET_COUNT 1

extern FILE * fileout;

const char * dataset_files[DATASET_COUNT][2] = {{"dataset/twitter-all.nodes", "dataset/twitter-all.edges"}};

int sum_array(int *a_in, int size)
{
    int sum = a_in[0];
    for(int i = 1 ; i < size ; i++)
        sum += a_in[i];
    return sum;
}

void T_BM_bfs(graph g_h, int source)
{
    /* necessary but not useful variables */
    #ifndef DP
    int one = 1, zero = 0;
    #endif

    /* set and define desicion variables */
    #ifndef DP
    int level = 0, workset_size = 1;
    #endif
    int covering_block_count = (g_h.size - 1)/COVERING_THREAD_PER_BLOCK + 1;
    int update_size = covering_block_count * COVERING_THREAD_PER_BLOCK;

    /* set reduction add kernel variables */
    int add_half_full_flag = covering_block_count%2;
    int add_block_count, add_block_size;
    if(add_half_full_flag){
        add_block_size = COVERING_THREAD_PER_BLOCK/2;
        add_block_count = covering_block_count;
    }else{
        add_block_size = COVERING_THREAD_PER_BLOCK;
        add_block_count = covering_block_count/2;
    }
    #ifndef DP
    int shared_size = add_block_size * sizeof(int);
    int * add_result_h = (int *)malloc(sizeof(int)*add_block_count);
    #endif

    /* initial graph on device based on BFS */
    graph g_d = consturct_graph_device(g_h);
    CUDA_CHECK_RETURN(cudaMalloc((void **)&(g_d.node_level_vector), sizeof(int)*g_h.size));
    CUDA_CHECK_RETURN(cudaMemset(g_d.node_level_vector, 20000, sizeof(int)*g_h.size));
    
    #ifdef DEBUG
    fprintf(fileout, "[DEBUG][INT_MAX] size of int in cpu: %d\n", sizeof(int));
    fprintf(fileout, "[DEBUG][INT_MAX] levels initialed with value of : 20000\n");
    fprintf(fileout, "[DEBUG][T_BM_BFS] graph successfully initialed on device\n");
    #endif

    /* initial arrays on device */
    char * update_d, * bitmap_d;
    CUDA_CHECK_RETURN(cudaMalloc((void **)&update_d, sizeof(char)*update_size));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&bitmap_d, sizeof(char)*update_size));
    CUDA_CHECK_RETURN(cudaMemset(update_d, 0, sizeof(char)*update_size));
    CUDA_CHECK_RETURN(cudaMemset(bitmap_d, 0, sizeof(char)*update_size));

    int * add_result_d;
    CUDA_CHECK_RETURN(cudaMalloc((void **)&(add_result_d), sizeof(int) * add_block_count));

    #ifdef DEBUG
    fprintf(fileout, "[DEBUG][T_BM_BFS] arrays successfully initialed on device\n");
    #endif

    #ifndef DP
    /* bfs first move in butmap and level vector */
    //TODO: use cudaMemset instead of copy or a better way
    CUDA_CHECK_RETURN(cudaMemcpy(&bitmap_d[source], &one, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(&g_d.node_level_vector[source], &zero, sizeof(int), cudaMemcpyHostToDevice));
    #endif

    #ifdef DEBUG
    fprintf(fileout, "[DEBUG][T_BM_BFS] first manual bfs move successfully done\n");
    #endif

    #ifndef DP
    while(workset_size != 0){
        one_bfs_T_BM<<<covering_block_count, COVERING_THREAD_PER_BLOCK>>>(g_d, bitmap_d, update_d, ++level);
        workset_update_BM<<<covering_block_count, COVERING_THREAD_PER_BLOCK>>>(update_d, bitmap_d);

        #ifdef DEBUG
        fprintf(fileout, "[DEBUG][T_BM_BFS] bfs kernels of level:%d is launched\n", level);
        #endif

        if(add_half_full_flag){
            add_kernel_half<<<add_block_count, add_block_size, shared_size>>>(bitmap_d, add_result_d);
        }else{
            add_kernel_full<<<add_block_count, add_block_size, shared_size>>>(bitmap_d, add_result_d);
        }

        CUDA_CHECK_RETURN(cudaDeviceSynchronize()); //wait for GPU
        CUDA_CHECK_RETURN(cudaGetLastError());

        CUDA_CHECK_RETURN(cudaMemcpy(add_result_h, add_result_d, add_block_count*sizeof(int), cudaMemcpyDeviceToHost));
        workset_size = sum_array(add_result_h, add_block_count);

        #ifdef DEBUG
        fprintf(fileout, "[DEBUG][T_BM_BFS] workset_size = %d\n", workset_size);
        #endif
    }
    #else
    argument argument_d;
    argument_d.covering_block_count = covering_block_count;
    argument_d.covering_block_size = COVERING_THREAD_PER_BLOCK;
    argument_d.add_half_full_flag = add_half_full_flag;
    argument_d.add_block_count = add_block_count;
    argument_d.add_block_size = add_block_size;
    argument_d.add_result = add_result_d;

    T_BM_bfs<<<1, 1>>>(g_d, source, bitmap_d, update_d, argument_d);

    CUDA_CHECK_RETURN(cudaDeviceSynchronize()); //wait for GPU
    CUDA_CHECK_RETURN(cudaGetLastError());
    #endif

    /* return level array of graph to host */
    CUDA_CHECK_RETURN(cudaMemcpy(g_h.node_level_vector, g_d.node_level_vector, sizeof(int)*g_h.size, cudaMemcpyDeviceToHost));

    #ifdef DEBUG
    fprintf(fileout, "[DEBUG][T_BM_BFS] node level vector successfully returned to CPU\n");
    #endif

    /* free memory GPU */
    destroy_graph_device(g_d);
    cudaFree(g_d.node_level_vector);
    cudaFree(update_d);
    cudaFree(bitmap_d);
    cudaFree(add_result_d);

    /* free memory CPU */
    #ifndef DP
    free(add_result_h);
    #endif
}

void adaptive_bfs(graph g_h, int source)
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
    CUDA_CHECK_RETURN(cudaMemset(g_d.node_level_vector, INT_MAX, sizeof(int)*g_h.size));
    
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
    initial_fileout();
    fprintf(fileout, "[MAIN] app.cu main\tDataset index: %d\n", DATASET_INDEX);

    /* read data set */
    graph g_h = consturct_graph(dataset_files[DATASET_INDEX][0], dataset_files[DATASET_INDEX][1]);

    /* initial bfs arrays */
    g_h.node_level_vector = (int *)malloc(sizeof(int)*g_h.size);

    #ifdef DEBUG
    fprintf(fileout, "[DEBUG][MAIN] running sequential bfs with graph size: %d\n", g_h.size);
    #endif

    /* sequentinal run */
    set_clock();
    sequential_run_bfs_QU(&g_h, 0);
    double elapced = get_elapsed_time();

    fprintf(fileout, "[MAIN] returning sequential bfs, time: %.2f\n", elapced);

    #ifdef DEBUG
    fprintf(fileout, "[DEBUG] first 10 nodes level (sequentianl):\n");
    for(int i=0;i<10;i++){
        fprintf(fileout, "node %d | level %d\n", i, g_h.node_level_vector[i]);
    }
    #endif

    /* Save sequential result for future use */
    int * sequential_result = (int *)malloc(sizeof(int)*g_h.size);
    memcpy(sequential_result, g_h.node_level_vector, sizeof(int)*g_h.size);

    #ifdef DEBUG
    fprintf(fileout, "[DEBUG] first 10 sequential result:\n");
    for(int i=0;i<10;i++){
        fprintf(fileout, "node %d | level %d\n", i, g_h.node_level_vector[i]);
    }
    #endif

    /* parallel run (T_BM) */
    set_clock();
    T_BM_bfs(g_h, 0);
    elapced = get_elapsed_time();

    fprintf(fileout, "[MAIN] returning parallel (T_BM) bfs, time: %.2f\n", elapced);

    #ifdef DEBUG
    fprintf(fileout, "[DEBUG] first 10 nodes level (parallel):\n");
    for(int i=0;i<10;i++){
        fprintf(fileout, "node %d | level %d\n", i, g_h.node_level_vector[i]);
    }
    #endif

    /* make compare files */
    make_compare_file("out/compare_seq_TBM.out", "sequentinal", sequential_result, "T_BM", g_h.node_level_vector, g_h.size);

    /* free allocated memory in main function */
    free(g_h.node_level_vector);
    destroy_graph(g_h);

    return 0;
}
#else

int main(int argc, char * argv[])
{
    int a[5] = {5, 3, 2, 1, 9};
    int b[5];

    memcpy(b, a, sizeof(a));

    for(int i=0;i<5;i++){
        printf("b[%d]=%d\t", i, b[i]);
    }

    return 0;
}
#endif
