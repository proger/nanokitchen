
import torch
from torch.utils.cpp_extension import load_inline
from pathlib import Path

cuda_source = (Path(__file__).parent / 'cuscan.cuh').read_text()

cpp_source = """
std::vector<at::Tensor> simple_scan_forward(const at::Tensor &gates, const at::Tensor &tokens);
"""

module = load_inline(
    name='cuscan',
    cpp_sources=[cpp_source],
    cuda_sources=[cuda_source],
    functions=['simple_scan_forward'],
    verbose=True
)
simple_scan_forward = module.simple_scan_forward

if __name__ == '__main__':
    print(module.simple_scan_forward(torch.ones(1,1,128).cuda(), torch.ones(1,1,128).cuda()))