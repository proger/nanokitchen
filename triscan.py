import torch
import triton
import triton.language as tl


@triton.jit
def scan1(
    gates,
    tokens,
    outputs,
    SEQUENCE_LENGTH: tl.constexpr,
):
    out = 0.
    for i in range(SEQUENCE_LENGTH):
        if i == 0:
            out = tl.load(tokens + i)
        else:
            out = tl.load(gates + i) * out + tl.load(tokens + i)
        tl.store(outputs + i, out)

@triton.jit
def scan2(
    gates,
    tokens,
    outputs,
    SEQUENCE_LENGTH: tl.constexpr,
):
    out = 0.
    for i in tl.static_range(SEQUENCE_LENGTH):
        if i == 0:
            out = tl.load(tokens + i)
        else:
            out = tl.load(gates + i) * out + tl.load(tokens + i)
        tl.store(outputs + i, out)


@triton.testing.perf_report([
    triton.testing.Benchmark(
        x_names=["SEQUENCE_LENGTH"],  # argument names to use as an x-axis for the plot
        x_vals=[2**i for i in range(17)],
        line_arg="provider",  # argument name whose value corresponds to a different line in the plot
        line_vals=["scan1", "scan2", "eager"],  # argument values to use as different lines in the plot
        line_names=["scan1", "scan2", "eager"],  # legend entries to use for each line
        plot_name="scan1",  # name of the plot
        args={
            #"SEQUENCE_LENGTH": seq_len,
        }
    ),
    # triton.testing.Benchmark(
    #     x_names=["SEQUENCE_LENGTH"],
    #     x_vals=[2**i for i in range(9)], # short sequence lengths as loop compilation is very slow
    #     line_arg="provider",
    #     line_vals=["scan1", "eager", "compile"],
    #     line_names=["scan1", "eager", "compile"],
    #     plot_name="scan-compiled",
    #     args={
    #         #"SEQUENCE_LENGTH": seq_len,
    #     }
    # ),
])
def bench(provider, SEQUENCE_LENGTH, device="cuda"):
    B, C, T = 1, 1, SEQUENCE_LENGTH
    gates, tokens = init(B, C, T, device)
    outputs = torch.empty_like(tokens)

    match provider:
        case "scan1":
            scan = lambda: scan1[(SEQUENCE_LENGTH,)](gates, tokens, outputs, SEQUENCE_LENGTH)
        case "scan2":
            scan = lambda: scan2[(SEQUENCE_LENGTH,)](gates, tokens, outputs, SEQUENCE_LENGTH)
        case "eager":
            scan = lambda: scan_eager(gates, tokens, outputs)
        case "compile":
            # compilation times are proportional to sequence length!
            compiled = torch.compile(scan_eager)
            scan = lambda: compiled(gates, tokens, outputs)
        case _:
            raise ValueError(f"Unknown provider {provider}")

    ms = triton.testing.do_bench(scan, warmup=25, rep=100)
    return ms


def init(B, C, T, device):
    gates = 0.999 + 0.001 * torch.rand(B, C, T, device=device)
    tokens = torch.rand(B, C, T, device=device)
    return gates, tokens


def scan_eager(gates, tokens, outputs):
    _, _, T = gates.shape
    for i in range(T):
        if i == 0:
            outputs[:, :, i] = tokens[:, :, i]
        else:
            outputs[:, :, i] = gates[:, :, i] * outputs[:, :, i-1] + tokens[:, :, i]


def test_allclose():
    device = 'cuda'
    B, C, T = 1, 1, 512
    gates, tokens = init(B, C, T, device)
    outputs = torch.empty_like(tokens)
    scan1[(T,)](gates, tokens, outputs, T)

    outputs2 = torch.empty_like(tokens)
    scan2[(T,)](gates, tokens, outputs2, T)

    assert torch.allclose(outputs, outputs2)

    outputs_naive = torch.empty_like(tokens)
    scan_eager(gates, tokens, outputs_naive)

    assert torch.allclose(outputs, outputs_naive)


if __name__ == '__main__':
    bench.run(save_path=".", print_data=True)