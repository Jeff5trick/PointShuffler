#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ATen.h>
#include <limits> 
#include <tuple>



void parallel_strided_sampling_kernel_wrapper(const int b, const int n, const int m, 
                                           const float *__restrict__ points, 
                                           int *__restrict__ order, 
                                           int *__restrict__ idxs, 
                                           float *__restrict__ out_coord,
                                           cudaStream_t stream);


std::tuple<torch::Tensor, torch::Tensor> parallel_strided_sampling(torch::Tensor points, torch::Tensor order, int m) {
    
    int b = points.size(0);
    int n = points.size(1);
    auto points_ptr = points.data_ptr<float>();
    auto order_ptr = order.data_ptr<int>();

    
    auto idxs = torch::empty({m}, torch::kInt32).to(points.device());
    auto out_coord = torch::empty({m, 3}, torch::kFloat32).to(points.device());

    
    parallel_strided_sampling_kernel_wrapper(b, n, m, points_ptr, 
                                          order_ptr, idxs.data_ptr<int>(), 
                                          out_coord.data_ptr<float>(),
                                          at::cuda::getCurrentCUDAStream());

    return std::make_tuple(idxs, out_coord);

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("parallel_strided_sampling", &parallel_strided_sampling, "Uniform Point Sampling (CUDA)");
}
