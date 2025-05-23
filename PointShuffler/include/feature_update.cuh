#include "common.cuh"

using data_type = float;
#define m_limit 8192

int feature_updata(const int n_points,  
        feat_setting setting,
        float *__restrict__ feats,  
        float *__restrict__ mlp_result,
        float *__restrict__ weight_1,
        float *__restrict__ weight_2,
        float *__restrict__ weight_3,
        cublasHandle_t cublasH,
        cudaStream_t feature_stream)
{

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;
    
 

    const int m = n_points; int n = setting.out_channel_1; int k = setting.in_channel;
    float *result1 = nullptr;
     
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&result1), sizeof(float) * m*n));
    
    bn_m_loader<<<1, 1024, 0, feature_stream>>>(m, n, setting.bn1_m, result1);

    CUBLAS_CHECK(
       cublasSgemm(cublasH, transa, transb, n, m, k, &setting.bn1_var, weight_1, n, feats, k, &setting.bn1_var, result1, n));
    
    cudaDeviceSynchronize();

 

    n = setting.out_channel_2;k = setting.out_channel_1;
    float *result2 = nullptr;
    
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&result2), sizeof(float) * m*n));
   
    bn_m_loader<<<1, 1024, 0, feature_stream>>>(m, n, setting.bn2_m, result2);
    
    CUBLAS_CHECK(
        cublasSgemm(cublasH, transa, transb, n, m, k, &setting.bn2_var, weight_2, n, result1, k, &setting.bn2_var, result2, n));
    cudaDeviceSynchronize();
    
    CUDA_CHECK(cudaFree(result1));



    n = setting.out_channel_3;k = setting.out_channel_2;
    
    bn_m_loader<<<1, 1024, 0, feature_stream>>>(m, n, setting.bn3_m, mlp_result);
    
    CUBLAS_CHECK(
        cublasSgemm(cublasH, transa, transb, n, m, k, &setting.bn3_var, weight_3, n, result2, k, &setting.bn3_var, mlp_result, n));

    cudaDeviceSynchronize();
    
    CUDA_CHECK(cudaFree(result2));


    return EXIT_SUCCESS;
}
