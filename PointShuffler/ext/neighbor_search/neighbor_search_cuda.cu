
#include "../common.cuh"
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void shared_mask_init_kernel(const int n, const int m, const int block_num,
                                    int *__restrict__ shared_len,
                                    bool *__restrict__ isn_shared, bool *__restrict__ have_center,int *__restrict__ neighbor_len)
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    for(int g=bid;g<block_num;g += gridDim.x)
    {
        int write_offset = g * n;
        for(int t=tid;t<n;t += blockDim.x)
        {
            isn_shared[write_offset + t] = false;
        }
        if(tid==0)
        {
            have_center[g] = false;
        }
    }
}

__global__ void shared_neighbor_search_kernel(const int n, const int m, const int block_num, const int shared_k,float r, \
                                   float *__restrict__ points, \
                                   int *__restrict__ center_index, \
                                   int *__restrict__ point2group, \
                                   bool *__restrict__ isn_shared,
                                   int *__restrict__ u_len, \
                                   int *__restrict__ u_offset, \
                                   int *__restrict__ u_order, \
                                   int *__restrict__ index,      
                                   float *__restrict__ distance,  
                                   const int search_total,
                                   int *__restrict__ searching_array,  
                                   int *__restrict__ searching_length, 
                                   int *__restrict__ searching_offset, 
                                   int *__restrict__ len_per_group, 
                                   bool *__restrict__ have_center,
                                   int *__restrict__ _1center_in_group,
                                   int *__restrict__ neighbor_len
                                   )
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;


    for(int i = bid; i < m; i += gridDim.x)
    { 
        int center = center_index[i];
        int u_group = point2group[center];
        have_center[u_group] = true;
        atomicExch(&_1center_in_group[u_group], i);
        
        int write_offset = searching_offset[i];

        float x_c = points[center * 3 + 0];
        float y_c = points[center * 3 + 1];
        float z_c = points[center * 3 + 2];

        for(int h = 0; h < len_per_group[u_group]; h++)
        {
            int group = searching_array[u_group * search_total + h];
            int len = u_len[group];
            int offset = u_offset[group];

            for(int j = tid; j < len; j += blockDim.x)
            {
                int sub_point = u_order[offset + j];

                float x_s = points[sub_point * 3 + 0];
                float y_s = points[sub_point * 3 + 1];
                float z_s = points[sub_point * 3 + 2];

                float dist = (x_c - x_s) * (x_c - x_s) + (y_c - y_s) * (y_c - y_s) + (z_c - z_s) * (z_c - z_s);
                 
                distance[write_offset + j] = dist;
                index[write_offset + j] = sub_point;
            }
            write_offset += len;
        }

        int c_len = searching_length[i];

        __syncthreads();

        int compare_len=0;
        
        int read_offset = searching_offset[i];
        int best_index;

        if(c_len < shared_k)
        { 
            compare_len = c_len;
            
            
        }
        else
        {
            compare_len = shared_k;

            for(int k = 0; k < compare_len; k++) 
            {
                best_index = c_len - 1; 

                for(int s = tid; s < c_len - k; s += blockDim.x)  
                { 
                    best_index = nearer_point(read_offset, s + k, best_index, distance);
                }

                __syncthreads();
                
                for(int offset = 16; offset > 0; offset /= 2)
                {
                    best_index = nearer_point(read_offset, __shfl_down_sync(0xffffffff, best_index, offset), best_index,distance);
                }
                
                if(tid == 0)
                {
                    float best = distance[read_offset + best_index];
                    distance[read_offset + best_index] = distance[read_offset + k];
                    distance[read_offset + k] = best;
                    
                    int best_idx = index[read_offset + best_index];
                    index[read_offset + best_index] = index[read_offset + k];
                    index[read_offset + k] = best_idx;
                    
                }
            }
        }


        __syncthreads();

        float dist =0.;
        neighbor_len[i] = 0;
        for(int p = tid; p < c_len; p += blockDim.x)
        {
            dist = distance[read_offset+p];
            if(dist>r || p>= compare_len)  
            {
                isn_shared[u_group * n + index[read_offset + p]] = true;
            }
            else
            {                
                atomicAdd(&neighbor_len[i], 1);
            }
        }
        
    }
    
}

__global__ void unique_neighbor_search_kernel(const int n, const int m, const int shared_k, float r, \
                                              int *__restrict__ center_index, \
                                              int *__restrict__ point2group, \
                                              bool *__restrict__ isn_shared, \
                                              int *__restrict__ index, \
                                              float *__restrict__ distance, \
                                              int *__restrict__ searching_length, \
                                              int *__restrict__ searching_offset, \
                                              int *__restrict__ shared_len,
                                              int *__restrict__ neighbor_len)
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    
    for (int i = bid; i < m; i += gridDim.x)
    {
        shared_len[i] = 0;
        int center = center_index[i];
        int u_group = point2group[center];

        int read_offset = searching_offset[i];
        int c_len = searching_length[i];
        int compare_len = min(c_len, shared_k);
        
         
        if(tid == 0) {
            for(int q = 0; q < neighbor_len[i]; ++q) {
                int read_idx = index[read_offset + q];
                if(!isn_shared[u_group * n + read_idx]) {
                    int offset = shared_len[i];
                    float tmp = distance[read_offset + q];
                    distance[read_offset + q] = distance[read_offset + offset];
                    distance[read_offset + offset] = tmp;
                    index[read_offset + q] = index[read_offset + offset];
                    index[read_offset + offset] = read_idx;
                    shared_len[i]++;
                }
            }
        }

    }
}

void neighbor_search_kernel_launcher(const int n, const int m, const int block_num, const int shared_k, const float r,\
    float *__restrict__ points, \
    int *__restrict__ center_index, \
    int *__restrict__ point2group, \
    int *__restrict__ u_len, \
    int *__restrict__ u_offset, \
    int *__restrict__ u_order, \
    bool *__restrict__ isn_shared, \
    int *__restrict__ index,      
    float *__restrict__ distance,  
    const int search_total,
    int *__restrict__ searching_array,
    int *__restrict__ searching_length,
    int *__restrict__ searching_offset,
    int *__restrict__ len_per_group,
    bool *__restrict__ have_center,
    int *__restrict__ _1center_in_group,
    int *__restrict__ shared_len,  
    int *__restrict__ neighbor_len,
    cudaStream_t stream)
{
    dim3 grid(m);
    dim3 block(32);  
    
    shared_mask_init_kernel<<<grid, block, shared_k * sizeof(int), stream>>>(n, m, block_num, shared_len, isn_shared, have_center,neighbor_len);

    shared_neighbor_search_kernel<<<grid, block, 0, stream>>>(n, m, block_num, shared_k,r, points, center_index, point2group,isn_shared, u_len, u_offset, u_order, index, distance, search_total, searching_array, searching_length, searching_offset, len_per_group, have_center, _1center_in_group,neighbor_len);
    
    unique_neighbor_search_kernel<<<grid, block, 0, stream>>>(n, m, shared_k, r,center_index, point2group, isn_shared, index, distance, searching_length, searching_offset, shared_len,neighbor_len);

}
