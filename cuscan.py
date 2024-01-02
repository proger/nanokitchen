
import torch
from torch.utils.cpp_extension import load_inline
from pathlib import Path

cuda_source = (Path(__file__).parent / 'cuscan.cuh').read_text()

cpp_source = """
std::vector<at::Tensor> simple_scan_forward(const at::Tensor &tokens, const at::Tensor &gates);
"""

module = load_inline(
    name='cuscan',
    cpp_sources=[cpp_source],
    cuda_sources=[cuda_source],
    functions=['simple_scan_forward'],
    verbose=True
)

print(module.simple_scan_forward(torch.ones(1,1,128).cuda(), torch.ones(1,1,128).cuda()))