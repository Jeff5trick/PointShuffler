#include "include/parallel_strided_sampling.cuh"
#include "include/partitioning.cuh"
#include "include/neighbor_search.cuh"
#include "include/feature_update.cuh"
#include "include/multi_hop.cuh"
#include "include/shared_aggregation.cuh"
#include "include/unique_aggregation.cuh"
#define point_cloud_file "../data/modelnet40_test.txt"


#define new_implement true
__global__ void writeToDeviceMemory(float *__restrict__ matrix, const int dim_1, const int dim_2); 

 
int main(int argc, char *argv[]) {
    cudaSetDevice(7);

    const int n=1024, m=512, K=32;
    float r = 0.2;
    const int block_size = 10;

    const int block_num = block_size*block_size*block_size; 
    float step = max_coordinate/block_size + correction_factor;
    const int hop=1;
    const int search_size = 2*hop+1;
    const int search_total = search_size*search_size*search_size;
    
    feat_setting setting;
    setting.in_channel = 3;setting.out_channel_1 = 64;setting.out_channel_2 = 64;setting.out_channel_3 = 128;
    setting.bn1_m = 1;setting.bn1_var=0.5;setting.bn2_m = 0.2;setting.bn2_var=0.5;setting.bn3_m = 0.3;setting.bn3_var=0.5;

    

    float h_weight_1[setting.in_channel][setting.out_channel_1];
    float h_weight_2[setting.out_channel_1][setting.out_channel_2];
    float h_weight_3[setting.out_channel_2][setting.out_channel_3];
    
    
    for(int i=0;i < setting.out_channel_1;i++)
        for(int j=0;j < setting.in_channel;j++) 
            h_weight_1[j][i] = i+1;
    
    for(int i=0;i < setting.out_channel_2;i++)
        for(int j=0;j < setting.out_channel_1;j++) 
            h_weight_2[j][i] = i+1;

    for(int i=0;i < setting.out_channel_3;i++)
        for(int j=0;j < setting.out_channel_2;j++) 
            h_weight_3[j][i] = i+1;

    setting.weight_1 = (float*)h_weight_1;
    setting.weight_2 = (float*)h_weight_2;
    setting.weight_3 = (float*)h_weight_3;

    float *d_weight_1 = nullptr;
    float *d_weight_2 = nullptr;
    float *d_weight_3 = nullptr;

    cudaStream_t load_stream = NULL;
    cudaStreamCreate(&load_stream);

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_weight_1), sizeof(float) * setting.in_channel*setting.out_channel_1));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_weight_2), sizeof(float) * setting.out_channel_1*setting.out_channel_2));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_weight_3), sizeof(float) * setting.out_channel_2*setting.out_channel_3));

    CUDA_CHECK(cudaMemcpyAsync(d_weight_1, h_weight_1, sizeof(float) * setting.in_channel*setting.out_channel_1, cudaMemcpyHostToDevice, load_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_weight_2, h_weight_2, sizeof(float) * setting.out_channel_1*setting.out_channel_2, cudaMemcpyHostToDevice, load_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_weight_3, h_weight_3, sizeof(float) * setting.out_channel_2*setting.out_channel_3, cudaMemcpyHostToDevice, load_stream));
    
    cudaStreamDestroy(load_stream);
    

    cudaStream_t stream = NULL;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));


    float h_c[n][3]; 
    float h_feat[n][setting.in_channel];
    
    int h_search_valid_length;

    read_point_cloud((float*)h_c, n,3, point_cloud_file);
    read_point_cloud((float*)h_feat, n,setting.in_channel,point_cloud_file);

    float *d_c = nullptr;
    float *d_feat = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c), sizeof(float) * n*3));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_feat), sizeof(float) * n*3));
    
    CUDA_CHECK(cudaMemcpyAsync(d_c, h_c, sizeof(float) * n*3, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_feat, h_feat, sizeof(float) * n*3, cudaMemcpyHostToDevice, stream));   


    coord_offset h_xyz_offset = {X_Offset, Y_Offset, Z_Offset};

    int *d_point2group = nullptr;
    int *d_u_len = nullptr;
    int *d_u_offset = nullptr;
    int *d_u_order = nullptr;

//preset Level 0 Index

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_point2group), sizeof(int) * n));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_u_len), sizeof(int) * block_num));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_u_offset), sizeof(int) * block_num));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_u_order), sizeof(int) * n));

    
    int *d_center = nullptr;
    float *d_out_coord = nullptr;
    int h_center[m];
    float *d_temp;
    int b = 16;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_temp), sizeof(float) * n * b ));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_center), sizeof(int) * m));        
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_out_coord), sizeof(float) * m*3));

    
    int *d_search_array = nullptr;
    int *d_search_length = nullptr;
    int *d_search_offset = nullptr;
    int *d_search_valid_length = nullptr;
    int *d_len_per_group = nullptr;

//preset Level 1 Index 

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_search_array), sizeof(int) * block_num*search_total));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_search_length), sizeof(int) * m));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_search_offset), sizeof(int) * m));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_search_valid_length), sizeof(int) *1));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_len_per_group), sizeof(int) * block_num));


    float *d_mlp_result = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_mlp_result), sizeof(float) * n*setting.out_channel_3));
 
    bool *d_isn_shared = nullptr;
    bool *d_have_center = nullptr;
    int *d_1center_in_group = nullptr;
    int *d_shared_count = nullptr;
    int *d_neighbor_len = nullptr;

    //preset Level 2 Index

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_isn_shared), sizeof(bool) * block_num*n));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_have_center), sizeof(bool) * block_num));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_1center_in_group), sizeof(int) * block_num));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_shared_count), sizeof(int) * m));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_neighbor_len), sizeof(int) * m));
 
    float *d_gather_result = nullptr;
    
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_gather_result), sizeof(float) * block_num*setting.out_channel_3));

    float *d_out_points = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_out_points), sizeof(float) * m*setting.out_channel_3));

    cublasHandle_t cublasH = NULL;
    cudaStream_t feature_stream = NULL;
 

    int *d_ns_index = nullptr;
    float *d_ns_distance = nullptr;
    int h_point2group[n];        
    int h_search_length[m];
    int h_len_per_group[block_num];
    int h_search_array[block_num*search_total];
    int h_search_offset[m];
    float *d_result1 = nullptr;
    float *d_result2 = nullptr;

    // start process
 
    partition_kernel_launcher(n, step, block_num, block_size, h_xyz_offset, d_c, d_point2group, d_u_len, d_u_offset, d_u_order, stream);

    CUDA_CHECK(cudaMemcpyAsync(h_point2group, d_point2group, sizeof(int) * n, cudaMemcpyDeviceToHost, stream));

    parallel_strided_sampling_kernel_wrapper(b, n, m, d_c, d_u_order ,d_center,d_out_coord, stream);

    CUDA_CHECK(cudaMemcpyAsync(h_center, d_center, sizeof(int) * 1*m, cudaMemcpyDeviceToHost, stream));
    
    searching_array_kernel_launcher(block_size, block_num, hop, search_size, search_total, m, d_center, d_point2group, d_u_len, d_search_array, d_search_length, d_search_offset, d_len_per_group,d_search_valid_length, stream);

    CUDA_CHECK(cudaMemcpyAsync(&h_len_per_group, d_len_per_group, sizeof(int) * block_num, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(&h_search_length, d_search_length, sizeof(int) * m, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(&h_search_array, d_search_array, sizeof(int) * block_num*search_total, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(&h_search_offset, d_search_offset, sizeof(int) * m, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(&h_search_valid_length, d_search_valid_length, sizeof(int) * 1, cudaMemcpyDeviceToHost, stream));
    cudaDeviceSynchronize();

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_result1), sizeof(float) * n * setting.out_channel_1));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_result2), sizeof(float) * n * setting.out_channel_2));
    CUDA_CHECK(cudaStreamCreateWithFlags(&feature_stream, cudaStreamNonBlocking));


    CUBLAS_CHECK(cublasCreate(&cublasH));    
    CUBLAS_CHECK(cublasSetStream(cublasH, feature_stream));


    feature_updata(n, setting, d_feat, d_mlp_result, d_weight_1, d_weight_2, d_weight_3,cublasH, feature_stream);


    float *result = (float *)malloc(n * setting.out_channel_3 * sizeof(float));
    CUDA_CHECK(cudaMemcpyAsync(result, d_mlp_result, sizeof(float) * setting.out_channel_3*n, cudaMemcpyDeviceToHost,feature_stream));


 
    bool *h_isn_shared = (bool *)malloc(sizeof(bool) * block_num * n);
    int h_ns_index[h_search_valid_length];

    bool h_have_center[block_num];
    int h_1center_in_group[block_num];
    int h_shared_count[m];
    int h_neighbor_len[m];

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_ns_index), sizeof(int) * h_search_valid_length));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_ns_distance), sizeof(float) * h_search_valid_length));

    neighbor_search_kernel_launcher(n, m, block_num, K, r*r, d_c, d_center, d_point2group, d_u_len, d_u_offset, d_u_order, d_isn_shared, d_ns_index,d_ns_distance, search_total,d_search_array, d_search_length, d_search_offset, d_len_per_group, d_have_center, d_1center_in_group, d_shared_count, d_neighbor_len,stream);

    CUDA_CHECK(cudaMemcpyAsync(h_ns_index, d_ns_index, sizeof(int) * h_search_valid_length, cudaMemcpyDeviceToHost, stream));
 
    CUDA_CHECK(cudaMemcpyAsync(h_isn_shared, d_isn_shared, sizeof(bool) * block_num*n, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_have_center, d_have_center, sizeof(bool) * block_num, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_1center_in_group, d_1center_in_group, sizeof(int) * block_num, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_shared_count, d_shared_count, sizeof(int) * m, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_neighbor_len, d_neighbor_len, sizeof(int) * m, cudaMemcpyDeviceToHost, stream));

    
    shared_aggregation_kernel_launcher(n, block_num, setting.out_channel_3, K, d_center, d_have_center, d_isn_shared, d_mlp_result, d_search_length, d_search_offset, d_1center_in_group, d_ns_index, d_shared_count, d_gather_result, stream);

    unique_aggregation_kernel_launcher(n, m, setting.out_channel_3, K, d_center, d_point2group, d_mlp_result, d_ns_index, d_isn_shared, d_search_length, d_search_offset, d_gather_result, d_shared_count, d_out_points, stream);

}
 