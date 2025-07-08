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
        float *mlp_time,
        cublasHandle_t cublasH,
        cudaStream_t feature_stream)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;
    

    cudaEventRecord(start, feature_stream);

    const int m = n_points; int n = setting.out_channel_1; int k = setting.in_channel;
    float *result1 = nullptr;
    
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&result1), sizeof(float) * m*n));
    
    bn_m_loader<<<1, 1024, 0, feature_stream>>>(m, n, setting.bn1_m, result1);

    CUBLAS_CHECK(
       cublasSgemm(cublasH, transa, transb, n, m, k, &setting.bn1_var, weight_1, n, feats, k, &setting.bn1_var, result1, n));
    
    cudaEventRecord(stop, feature_stream);
    cudaEventSynchronize(stop);
    float first_mlp_time;
    cudaEventElapsedTime(&first_mlp_time, start, stop);

    cudaDeviceSynchronize();


    cudaEventRecord(start, feature_stream);

    n = setting.out_channel_2;k = setting.out_channel_1;
    float *result2 = nullptr;
    
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&result2), sizeof(float) * m*n));
   
    bn_m_loader<<<1, 1024, 0, feature_stream>>>(m, n, setting.bn2_m, result2);
    
    CUBLAS_CHECK(
        cublasSgemm(cublasH, transa, transb, n, m, k, &setting.bn2_var, weight_2, n, result1, k, &setting.bn2_var, result2, n));
    cudaDeviceSynchronize();
    
    CUDA_CHECK(cudaFree(result1));

    cudaEventRecord(stop, feature_stream);
    cudaEventSynchronize(stop);
    float second_mlp_time;
    cudaEventElapsedTime(&second_mlp_time, start, stop);


    cudaEventRecord(start, feature_stream);

    n = setting.out_channel_3;k = setting.out_channel_2;
    
    bn_m_loader<<<1, 1024, 0, feature_stream>>>(m, n, setting.bn3_m, mlp_result);
    
    CUBLAS_CHECK(
        cublasSgemm(cublasH, transa, transb, n, m, k, &setting.bn3_var, weight_3, n, result2, k, &setting.bn3_var, mlp_result, n));

    cudaDeviceSynchronize();
    
    CUDA_CHECK(cudaFree(result2));

    CUBLAS_CHECK(cublasDestroy(cublasH));

    cudaEventRecord(stop, feature_stream);
    cudaEventSynchronize(stop);
    float third_mlp_time;
    cudaEventElapsedTime(&third_mlp_time, start, stop);
    
    *mlp_time = first_mlp_time + second_mlp_time + third_mlp_time;

    return EXIT_SUCCESS;
}
