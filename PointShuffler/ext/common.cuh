#pragma once
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
using std::min;
using std::max;

#include "cublas_utils.h"
#include <ctime>
#include <random>
#include <fstream>
#include <sstream>
#include <string>


#define max_coordinate 3.
#define correction_factor 2e-6
#define TOTAL_THREADS 1024

#define point_cloud_file "/home/ty/code/point_test_re/points_select/data/s_0_3.txt"


#define X_Offset 9
#define Y_Offset 13.858
#define Z_Offset 0.93


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
    int pow = std::log2(static_cast<double>(work_size)); 
    *(num_turns) = pow;
    return std::max(std::min(1 << pow, TOTAL_THREADS), 1); 
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



void read_point_cloud(float* out, int n) {
    std::ifstream inFile(point_cloud_file);
    if (!inFile.is_open()) {
        std::cout << "Failed to open the file for reading." << std::endl;
        return;
    }

    std::string line;
    for (int i = 0; i < n; ++i) {
        if (std::getline(inFile, line)) {
            std::stringstream ss(line);
            std::string value;
            for (int j = 0; j < 4; ++j) {
                if (std::getline(ss, value, ',')) {
                    if (j < 3) {
                        *(out + i * 3 + j) = std::stof(value);
                    }
                }
            }
        }
    }

    inFile.close();
}


















inline __device__ int n_nearer_point(int index_offset, int index_point, int best_index, float *distances)
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