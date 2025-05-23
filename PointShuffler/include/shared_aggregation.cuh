#include "common.cuh"

__global__ void shared_aggregation_kernel(const int n, const int block_num, const int channel, const int shared_k,
                                        int *__restrict__ centers,
                                        bool *__restrict__ have_center,
                                        bool *__restrict__ isn_shared, 
                                        float *__restrict__ mlp_result, 
                                        int *__restrict__ searching_length, 
                                        int *__restrict__ searching_offset, 
                                        int *__restrict__ _1center_in_group,
                                        int *__restrict__ ns_index, 
                                        int *__restrict__ shared_len,  
                                        float *__restrict__ result) 
{
     
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
 
    
    for(int t=bid; t<block_num; t += gridDim.x) 
    {
        for(int i=tid; i<channel; i+=blockDim.x) 
        {
            if(have_center[t]) 
            {
                int index = _1center_in_group[t]; 
                int read_offset = searching_offset[index];
                float max_value = 0;
                for(int j=0; j<shared_len[index]; j++)
                {
                    int point = ns_index[read_offset + j];
                    max_value = max(max_value, mlp_result[point*channel + i]);
                }
                result[t*channel+i] = max_value;
                 
            }
            else
            {
                result[t*channel+i] = 0;
            }
        }
    }
}



void shared_aggregation_kernel_launcher(const int n, const int block_num, const int channel, const int shared_k,
                                        int *__restrict__ centers,
                                        bool *__restrict__ have_center,
                                        bool *__restrict__ isn_shared, 
                                        float *__restrict__ mlp_result, 
                                        int *__restrict__ searching_length, 
                                        int *__restrict__ searching_offset, 
                                        int *__restrict__ _1center_in_group,
                                        int *__restrict__ ns_index, 
                                        int *__restrict__ shared_len, 
                                        float *__restrict__ result, 
                                        cudaStream_t stream)
{
    dim3 grid(block_num); 
    dim3 block(channel); 
    if(block.x > 1024)
        block.x =1024;
    shared_aggregation_kernel<<<grid, block, 0, stream>>>(n, block_num, channel, shared_k, centers, have_center, isn_shared, mlp_result, searching_length, searching_offset, _1center_in_group, ns_index, shared_len, result);
}
