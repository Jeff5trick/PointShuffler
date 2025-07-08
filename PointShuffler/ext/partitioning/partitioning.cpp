#include <torch/extension.h>
#include <vector>
#include <cuda_runtime.h>
#include <tuple>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ATen.h>

struct coord_offset {
    float x_offset;
    float y_offset;
    float z_offset;
};

void parallel_strided_sampling_kernel_wrapper(const int n, float step, const int block_num, const int block_size, coord_offset xyz_offset,
                          torch::Tensor points,
                          torch::Tensor point2group,
                          torch::Tensor u_len,
                          torch::Tensor u_offset,
                          torch::Tensor u_order,
                          cudaStream_t stream);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> partitioning_forward(torch::Tensor points, float step, int block_size, coord_offset xyz_offset)  {
    int n = points.size(0);
    int block_num = block_size * block_size * block_size;

    auto options = torch::TensorOptions().dtype(torch::kInt32).device(points.device());
    torch::Tensor point2group = torch::zeros({n}, options);
    torch::Tensor u_len = torch::zeros({block_num}, options);
    torch::Tensor u_offset = torch::zeros({block_num}, options); 
    torch::Tensor u_order = torch::zeros({n}, options);

    parallel_strided_sampling_kernel_wrapper(n, step, block_num, block_size, xyz_offset, points, point2group, u_len, u_offset, u_order, at::cuda::getCurrentCUDAStream());
    return std::make_tuple(u_order, u_len, u_offset,point2group);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<coord_offset>(m, "coord_offset")
        .def(py::init<>())
        .def_readwrite("x_offset", &coord_offset::x_offset)
        .def_readwrite("y_offset", &coord_offset::y_offset)
        .def_readwrite("z_offset", &coord_offset::z_offset);

    m.def("partitioning", &partitioning_forward, "UGVI Forward (CUDA)")
    .def("partitioning", [](torch::Tensor points, float step, int block_size, coord_offset xyz_offset) {
        auto result = partitioning_forward(points, step, block_size, xyz_offset);
        return std::make_tuple(std::get<0>(result), std::get<1>(result), std::get<2>(result), std::get<3>(result));
    });
}