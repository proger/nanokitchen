
import torch
from torch.utils.cpp_extension import load_inline
from pathlib import Path

cuda_source = (Path(__file__).parent / 'warpscan1.cu').read_text()

cpp_source = """
at::Tensor warpscan_forward(const at::Tensor &gates, const at::Tensor &tokens);
"""

module = load_inline(
    name='warpscan',
    cpp_sources=[cpp_source],
    cuda_sources=[cuda_source],
    functions=['warpscan_forward'],
    verbose=True,
    extra_cuda_cflags=[
        "-O3",
        "-std=c++17",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "--ptxas-options=-v",
        "-lineinfo",
    ]
)
warpscan_forward = module.warpscan_forward

if __name__ == '__main__':
    #print(warpscan_forward(torch.ones(1,1,32).cuda(), torch.ones(1,1,32).cuda()))
    N = 2048
    g = torch.ones(1,1,N).cuda().float()
    t = torch.arange(N).cuda().view(1,1,N).float()
    print(warpscan_forward(g, t).long())
    print(t.cumsum(dim=-1).long())