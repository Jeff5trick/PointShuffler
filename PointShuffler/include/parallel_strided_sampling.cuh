#include "common.cuh"

__global__ void parallel_strided_sampling_kernel(const int b, const int n, const int m, const float *__restrict__ dataset, int *__restrict__ order, int *__restrict__ idxs, float *__restrict__ out_coord)
{
    if (m <= 0) return;
    int tid = threadIdx.x;
    float step = (float)n / m;
    for (int j = tid; j < m; j += blockDim.x) {
         
        int idx = min((int)(j * step), n - 1);
        idxs[j] = order[idx];  

         
        out_coord[j * 3] = dataset[order[idx] * 3];
        out_coord[j * 3 + 1] = dataset[order[idx] * 3 + 1];
        out_coord[j * 3 + 2] = dataset[order[idx] * 3 + 2];
    }
}

void parallel_strided_sampling_kernel_wrapper(const int b, const int n, const int m, const float *__restrict__ dataset, int *__restrict__ order, int *__restrict__ idxs, float *__restrict__ out_coord, cudaStream_t stream)
{
    int num_turns = 0;
    int n_threads = opt_n_threads(n, &num_turns);
    
    parallel_strided_sampling_kernel<<<1, n_threads, 0, stream>>>
    (b, n, m, dataset, order, idxs, out_coord);
}

