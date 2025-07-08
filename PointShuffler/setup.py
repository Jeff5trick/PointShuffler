from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

# Common compile arguments
common_compile_args = {
    "cxx": ["-O2", "-D_GLIBCXX_USE_CXX11_ABI=0"],
    "nvcc": ["-O2"]
}

setup(
    name="ext_PointShuffler",
    version="0.1",
    ext_modules=[
        CUDAExtension(
            name="partitioning",
            sources=[
                "ext/partitioning/partitioning.cpp",
                "ext/partitioning/partitioning_cuda.cu"
            ],
            extra_compile_args=common_compile_args
        ),
        CUDAExtension(
            name="sampling",
            sources=[
                "ext/sampling/parallel_strided_sampling.cpp",
                "ext/sampling/parallel_strided_sampling_cuda.cu"
            ],
            extra_compile_args=common_compile_args
        ),
        CUDAExtension(
            name="multi_hop_cuda",
            sources=[
                "ext/multi_hop/multi_hop_cuda.cu",
                "ext/multi_hop/multi_hop.cpp"
            ],
            extra_compile_args=common_compile_args
        ),
        CUDAExtension(
            name="neighbor_search",
            sources=[
                "ext/neighbor_search/neighbor_search_cuda.cu",
                "ext/neighbor_search/neighbor_search.cpp"
            ],
            extra_compile_args=common_compile_args
        ),
        CUDAExtension(
            name="shared_aggregation",
            sources=[
                "ext/shared_aggregation/shared_aggregation_cuda.cu",
                "ext/shared_aggregation/shared_aggregation.cpp"
            ],
            extra_compile_args=common_compile_args
        ),
        CUDAExtension(
            name="unique_aggregation",
            sources=[
                "ext/unique_aggregation/unique_aggregation_cuda.cu",
                "ext/unique_aggregation/unique_aggregation.cpp"
            ],
            extra_compile_args=common_compile_args
        ),
    ],
    cmdclass={"build_ext": BuildExtension}
)