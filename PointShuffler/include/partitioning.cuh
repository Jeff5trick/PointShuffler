#include "common.cuh"
#include<iostream>

typedef struct {
    float x_offset;
    float y_offset;
    float z_offset;
}coord_offset;

inline __device__ int get_group_with_offset(float x, float y, float z, float step, int block_size, coord_offset xyz_offset){
    int id_x=(x + xyz_offset.x_offset)/step;
    int id_y=(y + xyz_offset.y_offset)/step;
    int id_z=(z + xyz_offset.z_offset)/step;
    int group = id_x + id_y*block_size + id_z*block_size*block_size;
    return group;
}


__global__ void partition_kernel(const int n, float step, const int block_num, const int block_size, coord_offset xyz_offset,
                            float *__restrict__ points, 
                             
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
    const int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int warp_tid = tid & 0x1F; 
    int group=0;

     
    for(int i=tid;i<n;i += blockDim.x) 
    {
        float x = points[i*3];
        float y = points[i*3+1];
        float z = points[i*3+2];
        group = get_group_with_offset(x, y, z, step, block_size, xyz_offset);
         
        point2group[i]=group; 
        atomicAdd(&u_len[group],1);

    }
  
    __syncthreads();

    if (warp_id == 0) 
    {
        int offset = 0, temp = 0;
        for (int i = warp_tid; i < ((block_num + 31) / 32) * 32; i += 32) 
        {
            offset = warp_tid == 0 ? temp : 0;
            int len = i < block_num ? u_len[i] : 0;

            temp = warpPrefixSum(warp_tid, offset + len);
            if (i < block_num)
                u_offset[i] = temp - len;
                
            temp = __shfl_sync(0xffffffff, temp, 31);
        }
    }

    __syncthreads();

    for(int j=tid;j<n;j += blockDim.x)
    {
        int group = point2group[j];
        int offset = u_offset[group];
        int index = atomicAdd(&count[group], 1) + offset;
        u_order[index] = j;
    }


}


 
void partition_kernel_launcher(const int n, float step, const int block_num,const int block_size, coord_offset xyz_offset, 
                            float *__restrict__ points, \
                             
                            int *__restrict__ point2group, \
                            int *__restrict__ u_len, \
                            int *__restrict__ u_offset, \
                            int *__restrict__ u_order, \
                             
                             
                             
                            cudaStream_t stream)
{
     
    dim3 block(block_num * 32);  
    if (block.x > 1024)
        block.x = 1024;
    
     
    partition_kernel<<<1, block, (block_num)*sizeof(unsigned int), stream>>>(n, step, block_num, block_size, xyz_offset, points, point2group, u_order, u_len, u_offset);
     
     
     

 

 
 
    
 
 
 
 
 

 
 
 
 
 
 
 

 
 
 
 
 
 
 
 
}