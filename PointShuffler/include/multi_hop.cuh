#include "common.cuh"

 
__global__ void searching_array_kerenl(const int block_size, const int block_num,
                                       const int hop, const int search_size, const int search_total, 
                                       const int m,
                                       int *__restrict__ centers,
                                       int *__restrict__ point2group,
                                       int *__restrict__ u_len,
                                       int *__restrict__ searching_array,
                                       int *__restrict__ searching_length,
                                       int *__restrict__ searching_offset,
                                       int *__restrict__ len_per_group,
                                       int *valid_length)
{
    const int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int warp_tid = tid & 0x1F; 

    int search_group;    

    

    for(int t = tid;t < block_num;t += blockDim.x)
    {
        int coord_x = t % block_size;
        int coord_y = (t / block_size) % block_size;
        int coord_z = t / (block_size*block_size);
        int count = 0;
        for(int z=0;z<search_size;z++)
        {   
            int move_z = z - hop;
            if(((coord_z+move_z)<0) || ((coord_z+move_z)>(block_size-1)))
            {
                continue;
            }
            else
            {
                for(int y=0;y<search_size;y++)
                {
                    int move_y = y - hop;
                    if(((coord_y+move_y)<0) || ((coord_y+move_y)>(block_size-1)))
                    {
                        continue;
                    }
                    else
                    {
                        for(int x=0;x<search_size;x++)
                        {                            
                            int move_x = x - hop; 
                            if(((coord_x+move_x)<0) || ((coord_x+move_x)>(block_size-1)))
                            {
                                continue;
                            }
                            else
                            {
                                search_group = t + move_x + move_y*block_size + move_z*block_size*block_size;
                                searching_array[t*search_total+count] = search_group;
                                count ++;
                            }                            
                        }
                    }
                }
            }
        }

        len_per_group[t] = count;
                

        
    }


    __syncthreads();
 
    for(int i = tid;i < m;i += blockDim.x)
    {
        int center = centers[i];
        int u_group = point2group[center];
        searching_length[i] = 0;
        int group;
        for(int j = 0;j< len_per_group[u_group];j++)
        {
            group = searching_array[u_group * search_total + j];

            searching_length[i] += u_len[group];
        }
    }

    __syncthreads();

    if (warp_id == 0)
    {
        int offset = 0, temp = 0;
        
        for (int i = warp_tid; i < ((m + 31) / 32) * 32; i += 32) 
        {
            offset = warp_tid == 0 ? temp : 0;
            int len = i < m ? searching_length[i] : 0;
            temp = warpPrefixSum(warp_tid, offset + len);
            if (i < m)
                searching_offset[i] = temp - len;
            temp = __shfl_sync(0xffffffff, temp, 31); 
        }
        if (warp_tid == 0)
             
            *valid_length = temp;
    }

     

     
}



void searching_array_kernel_launcher(const int block_size, const int block_num,
                                       const int hop, const int search_size, const int search_total, 
                                       const int m,
                                       int *__restrict__ centers,
                                       int *__restrict__ point2group,
                                       int *__restrict__ u_len,
                                       int *__restrict__ searching_array,
                                       int *__restrict__ searching_length,
                                       int *__restrict__ searching_offset,  
                                       int *__restrict__ len_per_group,  
                                       int *valid_length,                             
                                       cudaStream_t stream)
{
    dim3 block(block_num); 
    if(block.x > 1024)
        block.x =1024;

    searching_array_kerenl<<<1, block, 0, stream>>>(block_size, block_num, hop, search_size, search_total, m, centers, point2group, u_len, searching_array, searching_length, searching_offset, len_per_group, valid_length);
}