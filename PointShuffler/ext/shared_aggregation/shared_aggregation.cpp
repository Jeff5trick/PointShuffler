#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ATen.h>
#include <limits> 


void shared_aggregation_kernel_launcher(
    const int n, const int block_num, const int channel, const int shared_k,
    int* centers, bool* have_center, float* mlp_result,
    int* searching_offset, int* _1center_in_group,
    int* ns_index, int* shared_len, float* result, cudaStream_t stream);

torch::Tensor shared_aggregation_forward(
    torch::Tensor centers,
    torch::Tensor have_center,
    torch::Tensor mlp_result,
    torch::Tensor searching_offset,
    torch::Tensor _1center_in_group,
    torch::Tensor ns_index,
    torch::Tensor shared_len,
    int shared_k,
    int block_num) {
    
    
    TORCH_CHECK(centers.device().is_cuda(), "centers must be a CUDA tensor");
    TORCH_CHECK(have_center.device().is_cuda(), "have_center must be a CUDA tensor");
    TORCH_CHECK(mlp_result.device().is_cuda(), "mlp_result must be a CUDA tensor");
    TORCH_CHECK(searching_offset.device().is_cuda(), "searching_offset must be a CUDA tensor");
    TORCH_CHECK(_1center_in_group.device().is_cuda(), "_1center_in_group must be a CUDA tensor");
    TORCH_CHECK(ns_index.device().is_cuda(), "ns_index must be a CUDA tensor");

    

    const int n = mlp_result.size(0);
    const int channel = mlp_result.size(1);

    
    
    
    auto result = torch::full({block_num, channel}, 
        -std::numeric_limits<float>::infinity(),
        torch::device(centers.device()).dtype(torch::kFloat32));


    shared_aggregation_kernel_launcher(
        n, block_num, channel, shared_k,
        centers.data_ptr<int>(),
        have_center.data_ptr<bool>(),
        mlp_result.data_ptr<float>(),
        searching_offset.data_ptr<int>(),
        _1center_in_group.data_ptr<int>(),
        ns_index.data_ptr<int>(),
        shared_len.data_ptr<int>(),
        result.data_ptr<float>(),
        c10::cuda::getCurrentCUDAStream()
    );

    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("shared_aggregation", &shared_aggregation_forward, "Gather forward (CUDA)");
}
