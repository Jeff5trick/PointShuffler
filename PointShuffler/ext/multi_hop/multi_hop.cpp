#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ATen.h>


void searching_array_kernel_launcher(const int block_size, const int block_num,
                                     const int hop, const int search_size, const int search_total, 
                                     const int m,
                                     int *centers,
                                     int *point2group,
                                     int *u_len,
                                     int *searching_array,
                                     int *searching_length,
                                     int *searching_offset,  
                                     int *len_per_group,  
                                     int *valid_length,                             
                                     cudaStream_t stream);


std::vector<torch::Tensor> searching_array_kernel(
    int block_size, int block_num, int hop, int search_size, int search_total, int m,
    torch::Tensor centers, torch::Tensor point2group, torch::Tensor u_len)
{
    
    centers = centers.contiguous();
    point2group = point2group.contiguous();
    u_len = u_len.contiguous();

    
    auto centers_ptr = centers.data_ptr<int>();
    auto point2group_ptr = point2group.data_ptr<int>();
    auto u_len_ptr = u_len.data_ptr<int>();

    
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(centers.device());
    torch::Tensor searching_array = torch::zeros({block_num, search_total}, options);
    torch::Tensor searching_length = torch::zeros({m}, options);
    torch::Tensor searching_offset = torch::zeros({m}, options);
    torch::Tensor len_per_group = torch::zeros({block_num}, options);
    torch::Tensor valid_length = torch::zeros({1}, options);

    
    searching_array_kernel_launcher(
        block_size, block_num, hop, search_size, search_total, m,
        centers_ptr, point2group_ptr, u_len_ptr,
        searching_array.data_ptr<int>(), searching_length.data_ptr<int>(),
        searching_offset.data_ptr<int>(), len_per_group.data_ptr<int>(),
        valid_length.data_ptr<int>(),
        at::cuda::getCurrentCUDAStream()
    );

    
    return {searching_array, searching_length, searching_offset, len_per_group, valid_length};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("searching_array_kernel", &searching_array_kernel, "searching array kernel (CUDA)");
}
