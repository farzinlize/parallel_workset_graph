#include "kernels.cuh"
#include <stdio.h>
#include <limits.h>
#include "structures.h"
#include "sequential.h"
#include <limits.h>

extern "C"{
    #include "desicion_maker.h"
    #include "fuzzy_timing.h"
}

#define COVERING_THREAD_PER_BLOCK 1024
#define DATASET_COUNT 1

FILE * fileout;

const char * dataset_files[DATASET_COUNT][2] = {{"dataset/twitter-all.nodes", "dataset/twitter-all.edges"}};

void make_compare_file(char * file_name, char * result_1_name, int * result_1, char * result_2_name, int * result_2, int size)
{
    FILE * result_file = fopen(file_name, "wb");
    int max_level_1 = -1, max_level_2 = -1, fault_tree = 0, diffrenet = 0;
    int nodes_in_level_result_1[20] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    int nodes_in_level_result_2[20] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for(int i = 0; i<size; i++){
        if(result_1[i] >= 20 || result_2[i] >= 20){
            if((result_1[i] >= 20 && result_2[i] < 20) || (result_1[i] < 20 && result_2[i] >= 20))
                fault_tree++;
            fprintf(result_file, "node (is not in same tree): %d\t| %s level: %d\t| %s level: %d\n", i, result_1_name, result_1[i], result_2_name, result_2[i]);
            continue;
        }
        if(result_1[i] != result_2[i])    diffrenet++;
        if(result_1[i] > max_level_1)    max_level_1 = result_1[i];
        if(result_2[i] > max_level_2)   max_level_2 = result_2[i];
        nodes_in_level_result_1[result_1[i]]++;
        nodes_in_level_result_2[result_2[i]]++;
        fprintf(result_file, "node: %d\t| %s level: %d\t| %s level: %d\n", i, result_1_name, result_1[i], result_2_name, result_2[i]);
    }
    fprintf(result_file, "----------------------------------------\n");
    fprintf(result_file, "max level in %s run: %d\nmax level in %s run: %d\n", result_1_name, max_level_1, result_2_name, max_level_2);
    fprintf(result_file, "FAULT TREE --> %d\n", fault_tree);
    fprintf(result_file, "NUMBER OF DIFFRENT (between two run) --> %d\n", diffrenet);
    fprintf(result_file, "number of nodes in each level (%s)\n", result_1_name);
    for(int i=0;i<max_level_1;i++)    fprintf(result_file, "level: %d\tnumber of nodes: %d\n", i, nodes_in_level_result_1[i]);
    fprintf(result_file, "number of nodes in each level (%s)\n", result_2_name);
    for(int i=0;i<max_level_2;i++)    fprintf(result_file, "level: %d\tnumber of nodes: %d\n", i, nodes_in_level_result_2[i]);
    fclose(result_file);
}

void copy(int * a, int * b, int size)
{
    for(int i = 0; i < size ; i++){
        b[i] = a[i];
    }
}

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
    int one = 1, zero = 0;

    /* set and define desicion variables */
    int level = 0, workset_size = 1;
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
    int shared_size = add_block_size * sizeof(int);
    int * add_result_h = (int *)malloc(sizeof(int)*add_block_count);

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

    /* bfs first move in butmap and level vector */
    //TODO: use cudaMemset instead of copy or a better way
    CUDA_CHECK_RETURN(cudaMemcpy(&bitmap_d[source], &one, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(&g_d.node_level_vector[source], &zero, sizeof(int), cudaMemcpyHostToDevice));

    #ifdef DEBUG
    fprintf(fileout, "[DEBUG][T_BM_BFS] first manual bfs move successfully done\n");
    #endif

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
    free(add_result_h);
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
    #ifdef DEBUG
    fileout = fopen("out/ag_debug.out", "wb");
    #else
    fileout = fopen("out/ag.out", "wb");
    #endif

    fprintf(fileout, "[MAIN] app.cu main\tDataset index: %d\n", DATASET_INDEX);

    /* read data set */
    graph g_h = consturct_graph(dataset_files[DATASET_INDEX][0], dataset_files[DATASET_INDEX][1]);

    /* initial bfs arrays */
    g_h.node_level_vector = (int *)malloc(sizeof(int)*g_h.size);

    #ifdef DEBUG
    fprintf(fileout, "[DEBUG][MAIN] running sequential bfs with graph size: %d\n", g_h.size);
    #endif

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
    copy(g_h.node_level_vector, sequential_result, g_h.size);

    #ifdef DEBUG
    fprintf(fileout, "[DEBUG] first 10 sequential result:\n");
    for(int i=0;i<10;i++){
        fprintf(fileout, "node %d | level %d\n", i, g_h.node_level_vector[i]);
    }
    #endif

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

    make_compare_file("out/compare_seq_TBM.out", "sequentinal", sequential_result, "T_BM", g_h.node_level_vector, g_h.size);

    free(g_h.node_level_vector);
    destroy_graph(g_h);

    return 0;
}
#else

__global__ void testKernel(queue * q)
{
    fprintf(fileout, "queue size at the beginning of kernel: %d\n", q->size);

    for(int i=0;i<q->size;i++)
        fprintf(fileout, "items id: %d\titem value: %d\n", i, q->items[i]);

    q->size = 2;

    fprintf(fileout, "queue size at the end of kernel: %d\n", q->size);
}

int main(int argc, char * argv[])
{
    fprintf(fileout, "[MAIN] test main at app.cu\tDataset index: %d\n", DATASET_INDEX);

    queue * test;
    fprintf(fileout, "1\n");
    cudaMallocManaged(&test, sizeof(queue));
    cudaMallocManaged(&test->items, sizeof(int)*20);
    fprintf(fileout, "2\n");
    test->items[0] = 85;
    fprintf(fileout, "2.5\n");
    test->items[1] = 95;
    test->items[2] = 29;
    test->items[3] = 55;
    test->items[4] = 33;

    fprintf(fileout, "3\n");
    test->size = 5;

    fprintf(fileout, "4\n");
    testKernel<<<1, 1>>>(test);

    cudaDeviceSynchronize();
    
    fprintf(fileout, "queue size after kernel in main: %d\n", test->size);

    cudaFree(test->items);

    return 0;
}
#endif
