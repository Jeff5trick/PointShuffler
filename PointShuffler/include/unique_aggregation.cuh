#include "common.cuh"

__global__ void unique_aggregation_kernel(const int n, const int m, const int channel, const int shared_k,
                               int *__restrict__ centers,
                               int *__restrict__ point2group,
                               float *__restrict__ mlp_result, 
                               int *__restrict__ ns_index, 
                               bool *__restrict__ isn_shared,
                               int *__restrict__ searching_length, 
                               int *__restrict__ searching_offset, 
                               float *__restrict__ gather_result,
                               int *__restrict__ shared_len,  
                               float *__restrict__ result)
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;

    
    for(int i=bid; i<m; i += gridDim.x) 
    {
        int center = centers[i];
        int u_group = point2group[center];
        int read_offset = searching_offset[i];
        int len = shared_len[i];

        for(int j=tid; j<channel; j += blockDim.x) 
        {
            float max_value = gather_result[u_group*channel+j]; 

            int neighbor_num = min(shared_k, searching_length[i]);

            for(int k=0; k< neighbor_num - len ;k++) 
            {
                int point = ns_index[read_offset + len + k];     
                   
                max_value = max(max_value, mlp_result[point*channel+j]); 
            }
             
            result[i*channel+j] = max_value - mlp_result[center*channel+j];
             
        }
    }
    

}

void unique_aggregation_kernel_launcher(const int n, const int m, const int channel, const int shared_k,
                            int *__restrict__ centers,
                            int *__restrict__ point2group,
                            float *__restrict__ mlp_result, 
                            int *__restrict__ ns_index, 
                            bool *__restrict__ isn_shared,
                            int *__restrict__ searching_length, 
                            int *__restrict__ searching_offset, 
                            float *__restrict__ gather_result,
                            int *__restrict__ shared_len, 
                            float *__restrict__ result,
                            cudaStream_t stream)
{
    dim3 grid(m); 
    dim3 block(channel); 
    if(block.x > 1024)
        block.x =1024;
    unique_aggregation_kernel<<<grid, block, n*sizeof(bool), stream>>>(n, m, channel, shared_k, centers, point2group, mlp_result, ns_index, isn_shared, searching_length, searching_offset, gather_result, shared_len, result);
}



