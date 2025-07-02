#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ATen.h>


void neighbor_search_kernel_launcher(
    const int n, const int m, const int block_num, const int shared_k, const float r,
    float *points, int *center_index, int *point2group,
    int *u_len, int *u_offset, int *u_order,
    bool *isn_shared, int *ns_index, float *ns_distance, const int search_total,
    int *searching_array, int *searching_length, int *searching_offset,
    int *len_per_group, bool *have_center, int *_1center_in_group, int *shared_len,
    int *neighbor_len,  
    cudaStream_t stream);


std::vector<torch::Tensor> neighbor_search_forward(
    int n, int m, int block_num, int shared_k, float r,  
    torch::Tensor points, torch::Tensor center_index, torch::Tensor point2group,
    torch::Tensor u_len, torch::Tensor u_offset, torch::Tensor u_order,
    torch::Tensor searching_array, torch::Tensor searching_length, torch::Tensor searching_offset,
    torch::Tensor len_per_group, int valid_length, int search_total) {
    
    
    auto points_ptr = points.data_ptr<float>();
    auto center_index_ptr = center_index.data_ptr<int>();
    auto point2group_ptr = point2group.data_ptr<int>();
    auto u_len_ptr = u_len.data_ptr<int>();
    auto u_offset_ptr = u_offset.data_ptr<int>();
    auto u_order_ptr = u_order.data_ptr<int>();

    
    
    
    auto searching_array_ptr = searching_array.data_ptr<int>();
    auto searching_length_ptr = searching_length.data_ptr<int>();
    auto searching_offset_ptr = searching_offset.data_ptr<int>();
    auto len_per_group_ptr = len_per_group.data_ptr<int>();

    
    auto options_bool = torch::TensorOptions().dtype(torch::kBool).device(points.device());
    auto options_int = torch::TensorOptions().dtype(torch::kInt32).device(points.device());
    auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(points.device());
    torch::Tensor have_center = torch::zeros({block_num}, options_bool);
    torch::Tensor _1center_in_group = torch::zeros({block_num}, options_int);
    torch::Tensor shared_len = torch::zeros({m}, options_int);
    torch::Tensor isn_shared = torch::zeros({block_num, n}, options_bool);
    torch::Tensor neighbor_len = torch::zeros({m}, options_int);  

    
    torch::Tensor ns_index = torch::empty({valid_length}, options_int);  
    torch::Tensor ns_distance = torch::empty({valid_length}, options_float);  

    
    auto have_center_ptr = have_center.data_ptr<bool>();
    auto _1center_in_group_ptr = _1center_in_group.data_ptr<int>();
    auto shared_len_ptr = shared_len.data_ptr<int>();
    auto isn_shared_ptr = isn_shared.data_ptr<bool>();
    auto ns_index_ptr = ns_index.data_ptr<int>();
    auto ns_distance_ptr = ns_distance.data_ptr<float>();
    auto neighbor_len_ptr = neighbor_len.data_ptr<int>();  

    
    neighbor_search_kernel_launcher(
        n, m, block_num, shared_k, r,  
        points_ptr, center_index_ptr, point2group_ptr,
        u_len_ptr, u_offset_ptr, u_order_ptr,
        isn_shared_ptr, ns_index_ptr, ns_distance_ptr, search_total,
        searching_array_ptr, searching_length_ptr, searching_offset_ptr,
        len_per_group_ptr, have_center_ptr, _1center_in_group_ptr, shared_len_ptr,
        neighbor_len_ptr,  
        at::cuda::getCurrentCUDAStream());

    
    return {have_center, _1center_in_group, shared_len, isn_shared, ns_index, ns_distance, neighbor_len};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("neighbor_search", &neighbor_search_forward, "Top-K Neighbor Search with Radius Kernel (CUDA)");
}