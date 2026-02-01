#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ATen.h>
#include <limits> 


void unique_aggregation_kernel_launcher(const int n, const int m,  const int channel, const int shared_k,
    int *__restrict__ centers,
    int *__restrict__ point2group,
    float *__restrict__ mlp_result,
    int *__restrict__ ns_index,
    bool *__restrict__ isn_agv,
    int *__restrict__ searching_length,
    int *__restrict__ searching_offset,
    float *__restrict__ gather_result,
    int *__restrict__ shared_len,
    float *__restrict__ result,
    cudaStream_t stream);

torch::Tensor unique_aggregation_forward(
    torch::Tensor centers,
    torch::Tensor point2group,
    torch::Tensor mlp_result,
    torch::Tensor ns_index,
    torch::Tensor isn_agv,
    torch::Tensor searching_length,
    torch::Tensor searching_offset,
    torch::Tensor gather_result,
    torch::Tensor shared_len,
    int shared_k) {
    
    TORCH_CHECK(centers.device().is_cuda(), "centers must be a CUDA tensor");
    TORCH_CHECK(point2group.device().is_cuda(), "point2group must be a CUDA tensor");
    TORCH_CHECK(mlp_result.device().is_cuda(), "mlp_result must be a CUDA tensor");
    TORCH_CHECK(ns_index.device().is_cuda(), "ns_index must be a CUDA tensor");
    TORCH_CHECK(isn_agv.device().is_cuda(), "isn_agv must be a CUDA tensor");
    TORCH_CHECK(searching_length.device().is_cuda(), "searching_length must be a CUDA tensor");
    TORCH_CHECK(searching_offset.device().is_cuda(), "searching_offset must be a CUDA tensor");
    TORCH_CHECK(gather_result.device().is_cuda(), "gather_result must be a CUDA tensor");
    TORCH_CHECK(shared_len.device().is_cuda(), "shared_len must be a CUDA tensor");

    const int n = mlp_result.size(0);
    const int channel = mlp_result.size(1);
    const int m = centers.size(0);
    


    auto result = torch::full({m, channel}, 
        -std::numeric_limits<float>::infinity(),
        torch::device(centers.device()).dtype(torch::kFloat32));
    
    unique_aggregation_kernel_launcher(n,m,channel,shared_k,
        centers.data_ptr<int>(),
        point2group.data_ptr<int>(),
        mlp_result.data_ptr<float>(),
        ns_index.data_ptr<int>(),
        isn_agv.data_ptr<bool>(),
        searching_length.data_ptr<int>(),
        searching_offset.data_ptr<int>(),
        gather_result.data_ptr<float>(),
        shared_len.data_ptr<int>(),
        result.data_ptr<float>(),
        at::cuda::getCurrentCUDAStream()
    );
    
    return result;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("unique_aggregation", &unique_aggregation_forward, "Gather forward (CUDA)");
}
