extern "C"{
    #include "structures.h"
    #include "desicion_maker.h"
    #include "kernels.h"
}

#define COVERING_THREAD_PER_BLOCK 1024

int sum_array(int *a_in, int size)
{
    int sum = a_in[0];
    for(int i = 1 ; i < size ; i++)
        sum += a_in[i];
    return sum;
}

void run_bfs(struct graph * g_h)
{
    /* initial data on host */
    struct queue * workset = construct_queue(g_h->size);
    g_h->node_level_vector[0] = 0;
    queue_push(workset, 0);

    /* set and define desicion variables */
    int level = 0, block_count, thread_per_block, workset_size = workset->size;
    double avrage_outdeg = get_average_out_deg(g_h);
    int algo = decide(avrage_outdeg, workset_size, &block_count, &thread_per_block);
    int covering_block_count = (g_h->size - 1)/COVERING_THREAD_PER_BLOCK + 1;
    int update_size = covering_block_count * COVERING_THREAD_PER_BLOCK;
    int * add_result_h;
    add_result_h = (int *)malloc(sizeof(int)*covering_block_count);
    
    /* initial on and transform data to device */
    /*    initial workset queue on device      */
    struct queue * workset_d;
    CUDA_CHECK_RETURN(cudaMalloc((void **)&workset_d, sizeof(struct queue *)));
    CUDA_CHECK_RETURN(cudaMalloc( (void **)&workset_d->items, g_h->size));
    /*    transform workset queue to device    */
    CUDA_CHECK_RETURN(cudaMemcpy(workset_d->items, workset->items, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(&workset_d->size, &workset->size, sizeof(int), cudaMemcpyHostToDevice));
    /*    initial graph on device    */
    struct graph * g_d;
    CUDA_CHECK_RETURN(cudaMalloc((void **)&g_d, sizeof(struct graph *)));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&g_d->size, sizeof(int)));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&g_d->node_vector, sizeof(int)*(g_h->size+1)));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&g_d->edge_vector, sizeof(int)*(g_h->node_vector[g->size])));
    //TODO: MACRO for initialing level or distance array for GPU
    CUDA_CHECK_RETURN(cudaMalloc((void **)&g_d->node_level_vector, sizeof(int)*g_h->size));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&g_d->node_distance_vector, sizeof(int)*g_h->size));
    /*    transform graph to device  */
    CUDA_CHECK_RETURN(cudaMemcpy(g_d->node_vector, g_h->node_vector, sizeof(int)*(g_h->size+1), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(g_d->edge_vector, g_h->edge_vector, sizeof(int)*(g_h->node_vector[g->size]), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(g_d->size, g_h->size, sizeof(int), cudaMemcpyHostToDevice));
    /*    initial arrays on device    */
    char * update_d, * bitmap_d;
    int * add_result_d;
    int one = 1;
    CUDA_CHECK_RETURN(cudaMalloc((void **)&update_d, sizeof(char)*update_size));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&bitmap_d, sizeof(char)*g_h->size));
    //TODO: initial zero on GPU update_d and bitmap_d (copy a zero-array to GPU or run a kernel for it)
    CUDA_CHECK_RETURN(cudaMalloc((void **)&add_result_d, sizeof(int)*covering_block_count));
    /*    transform bitmap to device    */
    CUDA_CHECK_RETURN(cudaMemcpy(bitmap_d, &one, sizeof(int), cudaMemcpyHostToDevice));

    while (workset_size != 0)
    {
        if (algo == B_QU)
        {
            workset_update_QU<<<covering_block_count, COVERING_THREAD_PER_BLOCK>>>(update_d, workset_d);
            one_bfs_B_QU<<<block_count, thread_per_block>>>(g_d, workset_d, update_d, level++);
        } else if (algo == B_BM) 
        {
            workset_update_BM<<<covering_block_count, COVERING_THREAD_PER_BLOCK>>>(update_d, bitmap_d);
            one_bfs_B_BM<<<block_count, thread_per_block>>>(g_d, bitmap_d, update_d, level++);
        } else if (algo == T_QU)
        {
            workset_update_QU<<<covering_block_count, COVERING_THREAD_PER_BLOCK>>>(update_d, workset_d);
            one_bfs_T_QU<<<block_count, thread_per_block>>>(g_d, workset_d, update_d, level++);
        } else if (algo == T_BM)
        {
            workset_update_BM<<<covering_block_count, COVERING_THREAD_PER_BLOCK>>>(update_d, bitmap_d);
            one_bfs_T_BM<<<block_count, thread_per_block>>>(g_d, bitmap_d, update_d, level++);
        }
        /* calculate workset size and decide the next move */
        add_kernel<<<covering_block_count, COVERING_THREAD_PER_BLOCK, sizeof(int)*COVERING_THREAD_PER_BLOCK>>>(update_d, add_result_d);

        CUDA_CHECK_RETURN(cudaDeviceSynchronize()); //wait for GPU
        CUDA_CHECK_RETURN(cudaGetLastError());
        
        CUDA_CHECK_RETURN(cudaMemcpy(add_result_h, add_result_d, sizeof(int)*covering_block_count, cudaMemcpyDeviceToHost));
        workset_size = sum_array(add_result_h, covering_block_count);

        algo = decide(avrage_outdeg, workset_size, &block_count, &thread_per_block);
    }

    /*    free memory CPU and GPU    */
    //TODO
}

int main()
{
    return 0;
}