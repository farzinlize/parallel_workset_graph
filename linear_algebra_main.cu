#include "linear_algebra_main.cuh"

/* ### MAIN ### */
void linear_algebra_bfs_vector(graph g_h, int source)
{
    /* define useful variables */
    int maximum_threads_in_block = 1024;
    int one = 1, zero = 0, max_depth = 10;

    /* initial graph on device based on BFS */
    graph g_d = consturct_graph_device(g_h);
    CUDA_CHECK_RETURN(cudaMalloc((void **)&(g_d.node_level_vector), sizeof(int)*g_h.size));
    CUDA_CHECK_RETURN(cudaMemset(g_d.node_level_vector, -1, sizeof(int)*g_h.size));

    /* set source node level to zero */
    CUDA_CHECK_RETURN(cudaMemcpy(&g_d.node_level_vector[source], &zero, sizeof(int), cudaMemcpyHostToDevice));

    /* kernel grids configuration variables */
    int covering_block_count = (g_h.size - 1)/maximum_threads_in_block + 1;
    int covering_nlock_count_for_warps = covering_block_count * 32;
    dim3 grid_dim_thread(covering_block_count, 1, 1);
    dim3 grid_dim_warp(covering_nlock_count_for_warps, 1, 1);
    dim3 block_dim(maximum_threads_in_block, 1, 1);

    /* set shared memory size */
    int shared_size = sizeof(int) * maximum_threads_in_block;

    /* initial GPU arrays */
    int * multiplier_d;
    int * working_array;
    CUDA_CHECK_RETURN(cudaMalloc((void **)&(multiplier_d), sizeof(int)*g_h.size));
    CUDA_CHECK_RETURN(cudaMemset(multiplier_d, 0, sizeof(int)*g_h.size));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&(working_array), sizeof(int)*g_h.size));

    /* set first multiplier vector */
    CUDA_CHECK_RETURN(cudaMemcpy(&multiplier_d[source], &one, sizeof(int), cudaMemcpyHostToDevice));

    #ifdef DETAIL
    FILE * mult_report;
    mult_report = fopen("out/mult_report.out", "wb");
    int * multiplier_h = (int *)malloc(sizeof(int) * g_h.size);
    int j;

    CUDA_CHECK_RETURN(cudaMemcpy(multiplier_h, multiplier_d, sizeof(int)*g_h.size, cudaMemcpyDeviceToHost));

    fprintf(mult_report, "pre product\n[");
    for(j=0;j<g_h.size;j++){
        fprintf(mult_report, "%d\t", multiplier_h[j]);
    }
    fprintf(mult_report, "]\n");
    #endif

    /* Kernel */
    int level = 0;
    for(int i=0;i<max_depth;i++){
        spmv_csr_vector_kernel<<<grid_dim_warp, block_dim, shared_size>>>(g_d, multiplier_d, working_array, ++level);
        set_result<<<grid_dim_thread, block_dim>>>(g_d, multiplier_d, working_array);

        #ifdef DEBUG
        printf("multiply number %d operating |\t", i);
        #endif
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());
        #ifdef DEBUG
        printf("done\n");
        #endif

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
}

void linear_algebra_bfs_scalar(graph g_h, int source)
{
    /* define useful variables */
    int maximum_threads_in_block = 1024;
    int one = 1, zero = 0, max_depth = 10;

    /* initial graph on device based on BFS */
    graph g_d = consturct_graph_device(g_h);
    CUDA_CHECK_RETURN(cudaMalloc((void **)&(g_d.node_level_vector), sizeof(int)*g_h.size));
    CUDA_CHECK_RETURN(cudaMemset(g_d.node_level_vector, -1, sizeof(int)*g_h.size));

    /* set source node level to zero */
    CUDA_CHECK_RETURN(cudaMemcpy(&g_d.node_level_vector[source], &zero, sizeof(int), cudaMemcpyHostToDevice));

    /* kernel grids configuration variables */
    int covering_block_count = (g_h.size - 1)/maximum_threads_in_block + 1;
    dim3 grid_dim(covering_block_count, 1, 1);
    dim3 block_dim(maximum_threads_in_block, 1, 1);

    int * multiplier_d;
    int * working_array;
    CUDA_CHECK_RETURN(cudaMalloc((void **)&(multiplier_d), sizeof(int)*g_h.size));
    CUDA_CHECK_RETURN(cudaMemset(multiplier_d, 0, sizeof(int)*g_h.size));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&(working_array), sizeof(int)*g_h.size));

    /* set first multiplier vector */
    CUDA_CHECK_RETURN(cudaMemcpy(&multiplier_d[source], &one, sizeof(int), cudaMemcpyHostToDevice));

    #ifdef DETAIL
    FILE * mult_report;
    mult_report = fopen("out/mult_report.out", "wb");
    int * multiplier_h = (int *)malloc(sizeof(int) * g_h.size);
    int j;

    CUDA_CHECK_RETURN(cudaMemcpy(multiplier_h, multiplier_d, sizeof(int)*g_h.size, cudaMemcpyDeviceToHost));

    fprintf(mult_report, "pre product\n[");
    for(j=0;j<g_h.size;j++){
        fprintf(mult_report, "%d\t", multiplier_h[j]);
    }
    fprintf(mult_report, "]\n");
    #endif

    /* Kernel */
    int level = 0;
    for(int i=0;i<max_depth;i++){
        spmv_csr_scalar_kernel<<<grid_dim, block_dim>>>(g_d, multiplier_d, working_array, ++level);
        set_result<<<grid_dim, block_dim>>>(g_d, multiplier_d, working_array);

        #ifdef DEBUG
        printf("multiply number %d operating |\t", i);
        #endif
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());
        #ifdef DEBUG
        printf("done\n");
        #endif

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
}

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

        #ifdef DEBUG
        printf("multiply number operating: %d |\t", i);
        #endif
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());
        #ifdef DEBUG
        printf("done\n");
        #endif

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
    /* call DP kernel */
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