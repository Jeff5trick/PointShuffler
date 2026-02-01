#pragma once
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <algorithm>
#include "cublas_utils.h"
#include <ctime>
#include <random>
#include <fstream>
#include <sstream>
#include <string>

#include <unordered_set>
#include <vector>


#define max_coordinate 2.
#define correction_factor 2e-6
#define TOTAL_THREADS 1024

#define X_Offset 1.0010000467300415
#define Y_Offset 1.0010000467300415
#define Z_Offset 1.0010000467300415 

typedef struct {  
  int index;
  float distance;
} ns_data;

typedef struct {  
  int in_channel;
  int out_channel_1;
  int out_channel_2;
  int out_channel_3;
  float bn1_m;
  float bn1_var;
  float bn2_m;
  float bn2_var;
  float bn3_m;
  float bn3_var;
  float *weight_1;
  float *weight_2;
  float *weight_3;
  float *result;
  
  
} feat_setting;

__inline__ __host__ __device__ void generate_integer_power_of_two(int length, int *result)
{
    int pow = std::log(static_cast<double>(length)) / std::log(2.0);
    *(result) = pow;
} 

template <typename T>
__inline__ __device__ T warpPrefixSum(int id, T count) {
  for (int i = 1; i < 32; i <<= 1) {
    T val = __shfl_up_sync(0xffffffff, count, i);
    if (id >= i)
      count += val;
  }
  return count;
}


inline __device__ int nearer_point(int index_offset, int index_point, int best_index, float *distances)
{
    
    float a_distance = distances[index_offset + index_point];
    
    float b_distance = distances[index_offset + best_index];

    
    if (a_distance > b_distance)
    {
        return best_index;
    }
    else
    {
        return index_point;
    }
}


inline int opt_n_threads(int work_size , int *num_turns) {
    int pow = std::log(static_cast<double>(work_size)) / std::log(2.0);
    *(num_turns) = pow;
    return max(min(1 << pow, TOTAL_THREADS), 1);
}

template <typename T>
__inline __device__ void topk_update(int tid, int step, int n, T *__restrict__ dists, int *__restrict__ dists_i)
{
    int other_idx = tid+step;
          
    if(other_idx < n)
    {
        float dd1 = dists[tid];
        int idx1 = dists_i[tid];
        
        float dd2 = dists[other_idx];        
        int idx2 = dists_i[other_idx];
        if(dd1<dd2)
        {
            dists[tid] = dd2;
            dists_i[tid] = idx2;
        }
        else
        {
            dists[tid] = dd1;
            dists_i[tid] = idx1;
        }
    }
}

__global__ void bn_m_loader(const int n, const int out_channel, const float avg, float *__restrict__ result)
{
    const int tid = threadIdx.x;
    const int warps_total = blockDim.x >> 5;
    int warp_id = tid >> 5;
    int warp_tid = tid & 0x1F;

    for(int i=warp_id;i<n;i += warps_total)
    {
        for(int j=warp_tid;j<out_channel;j += 32)
        {
            result[i*out_channel+j]= -avg;
        }
    }
}

void generate_random_point_cloud(float max, float min, int n, float* a)
{
    
    std::random_device rd;
    std::mt19937 gen(rd()); 
    std::uniform_real_distribution<float> dist(min, max); 

	for(int i=0;i<n;i++){
		for(int j=0;j<3;j++)
            *(a+i*3+j)=dist(gen);
	}

}

void save_point_cloud(int n, float* data)
{
    std::ofstream outFile("../points.txt");
    
    if (!outFile.is_open()) {
        std::cout << "Failed to open the file for writing." << std::endl;
    }
    
    
    for(int i=0;i<n;i++){
		for(int j=0;j<3;j++)
            outFile << *(data+i*3+j) << " ";
	
        outFile << std::endl;
    }

    outFile.close();
}



void read_point_cloud(float* out, int n, int columns, const std::string& file_path) {
    std::ifstream inFile(file_path);
    if (!inFile.is_open()) {
        std::cerr << "无法打开文件: " << file_path << std::endl;
        return;
    }

    std::string line;
    for (int i = 0; i < n; ++i) {
        if (!std::getline(inFile, line)) {
            std::cout << "文件提前结束或读取失败" << std::endl;
            break;
        }

        std::stringstream ss(line);
        for (int j = 0; j < columns; ++j) {
            float value;
            if (!(ss >> value)) {
                std::cerr << "无法读取第 " << i + 1 << " 点，第 " << j + 1
                          << " 列的值，文件: " << file_path << std::endl;
                inFile.close();
                return;
            }
            out[i * columns + j] = value;

            if (j < columns - 1) {
                char comma;
                if (!(ss >> comma) || comma != ',') {
                    std::cerr << "期望逗号，第 " << i + 1 << " 点，第 " << j + 1
                              << " 列，文件: " << file_path << std::endl;
                    inFile.close();
                    return;
                }
            }
        }
    }
    inFile.close();
}

__global__ void share_size_test_kernel(const int n, float *__restrict__ points)
{
    extern __shared__ float s_points[];
    const int tid = threadIdx.x;
    for(int i = tid;i<n;i += blockDim.x)
    {
        s_points[n*3]=points[n*3];
        s_points[n*3+1]=points[n*3+1];
        s_points[n*3+2]=points[n*3+2];
    }
}

void share_size_test_kernel_launcher(const int n, float *__restrict__ points, cudaStream_t stream)
{
    share_size_test_kernel<<<1, 1024, sizeof(float)*n*3, stream>>>(n, points);
}

void printMemoryUsage() {
    size_t freeMem, totalMem;
    cudaError_t err = cudaMemGetInfo(&freeMem, &totalMem);
    if (err != cudaSuccess) {
        std::cerr << "Error in cudaMemGetInfo: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    std::cout << "Total GPU memory: " << totalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Free GPU memory: " << freeMem / (1024 * 1024) << " MB" << std::endl;
}

size_t count_common_elements(const std::vector<int>& seq1, const std::vector<int>& seq2) {
    std::unordered_set<int> set1(seq1.begin(), seq1.end());
    std::unordered_set<int> set2(seq2.begin(), seq2.end());
    
    size_t count = 0;
    for (int num : set1) {
        if (set2.count(num)) {
            count++;
        }
    }
    return count;
}