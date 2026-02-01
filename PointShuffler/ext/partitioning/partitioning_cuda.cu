#include "../common.cuh"
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

typedef struct {
    float x_offset;
    float y_offset;
    float z_offset;
} coord_offset;

inline __device__ int get_group_with_offset(float x, float y, float z, float step, int block_size, coord_offset xyz_offset){
    int id_x = (x + xyz_offset.x_offset) / step;
    int id_y = (y + xyz_offset.y_offset) / step;
    int id_z = (z + xyz_offset.z_offset) / step;
    int group = id_x + id_y * block_size + id_z * block_size * block_size;
    return group;
}

__global__ void parallel_strided_sampling_kernel(const int n, float step, const int block_num, const int block_size, coord_offset xyz_offset,
                            const float *__restrict__ points, 
                            int *__restrict__ point2group,
                            int *__restrict__ u_order,
                            int *__restrict__ u_len,
                            int *__restrict__ u_offset)
{
    extern __shared__ unsigned int temp_base[];
    unsigned int *count = temp_base;

    for (int i = threadIdx.x; i < block_num; i += blockDim.x) {
        count[i] = 0;
        u_len[i] = 0;
    }

    __syncthreads();

    int tid = threadIdx.x;
    for(int i = tid; i < n; i += blockDim.x)
    {
        float x = points[i * 3];
        float y = points[i * 3 + 1];
        float z = points[i * 3 + 2];
        int group = get_group_with_offset(x, y, z, step, block_size, xyz_offset);
        point2group[i] = group;
        atomicAdd(&u_len[group], 1);
    }

    __syncthreads();

    if (tid == 0) {
        int offset = 0;
        for (int i = 0; i < block_num; i++) {
            u_offset[i] = offset;
            offset += u_len[i];
        }
    }

    __syncthreads();

    for(int j = tid; j < n; j += blockDim.x)
    {
        int group = point2group[j];
        int offset = u_offset[group];
        int index = atomicAdd(&count[group], 1) + offset;
        u_order[index] = j;
    }
}

void parallel_strided_sampling_kernel_wrapper(const int n, float step, const int block_num, const int block_size, coord_offset xyz_offset, 
                          torch::Tensor points,
                          torch::Tensor point2group,
                          torch::Tensor u_len,
                          torch::Tensor u_offset,
                          torch::Tensor u_order,
                          cudaStream_t stream)
{
    const int threads = 1024;
    parallel_strided_sampling_kernel<<<1, threads, block_num * sizeof(unsigned int), stream>>>(n, step, block_num, block_size, xyz_offset,
                                                                          points.data_ptr<float>(),
                                                                          point2group.data_ptr<int>(),
                                                                          u_order.data_ptr<int>(),
                                                                          u_len.data_ptr<int>(),
                                                                          u_offset.data_ptr<int>());
}
