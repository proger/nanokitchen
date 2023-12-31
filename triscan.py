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
    global_id = tl.num_programs(axis=1) * tl.program_id(axis=0) + tl.program_id(axis=1)
    offset = global_id * SEQUENCE_LENGTH

    out = 0.
    for i in range(SEQUENCE_LENGTH):
        if i == 0:
            out = tl.load(tokens + offset + i)
        else:
            out = tl.load(gates + offset + i) * out + tl.load(tokens + offset + i)
        tl.store(outputs + offset + i, out)

@triton.jit
def scan2(
    gates,
    tokens,
    outputs,
    SEQUENCE_LENGTH: tl.constexpr,
    UNROLL_LENGTH: tl.constexpr,
):
    global_id = tl.num_programs(axis=1) * tl.program_id(axis=0) + tl.program_id(axis=1)
    offset = global_id * SEQUENCE_LENGTH

    out = 0.
    for chunk_i in range(tl.cdiv(SEQUENCE_LENGTH, UNROLL_LENGTH)):
        for unroll_i in tl.static_range(UNROLL_LENGTH):
            i = chunk_i * UNROLL_LENGTH + unroll_i
            if i == 0:
                out = tl.load(tokens + offset + i)
            else:
                out = tl.load(gates + offset + i) * out + tl.load(tokens + offset + i)
            tl.store(outputs + offset + i, out)

@triton.jit
def scan2_backward(
    gates,
    tokens,
    outputs,
    SEQUENCE_LENGTH: tl.constexpr,
    UNROLL_LENGTH: tl.constexpr,
):
    global_id = tl.num_programs(axis=1) * tl.program_id(axis=0) + tl.program_id(axis=1)
    offset = global_id * SEQUENCE_LENGTH

    out = 0.
    for chunk_i in range(tl.cdiv(SEQUENCE_LENGTH, UNROLL_LENGTH)-1, -1, -1):
        for unroll_i in tl.static_range(UNROLL_LENGTH-1, -1, -1):
            i = chunk_i * UNROLL_LENGTH + unroll_i
            if i == SEQUENCE_LENGTH-1:
                out = tl.load(tokens + offset + i)
            else:
                out = tl.load(gates + offset + i + 1) * out + tl.load(tokens + offset + i)
            tl.store(outputs + offset + i, out)

# from https://github.com/openai/triton/issues/2359
@triton.jit
def unpack64(merged):
    tl.static_assert(merged.dtype == tl.int64)
    b = (merged & 0xFFFFFFFF).to(tl.int32).to(tl.float32, bitcast=True)
    a = (merged >> 32).to(tl.int32).to(tl.float32, bitcast=True)
    return a, b

@triton.jit
def pack64(a, b):
    tl.static_assert(a.dtype == tl.float32)
    tl.static_assert(b.dtype == tl.float32)
    a = a.to(dtype=tl.int32, bitcast=True).to(tl.int64)
    a = a << 32  # shifted by 32 bits
    b = b.to(dtype=tl.int32, bitcast=True).to(tl.int64)
    return a | b

@triton.jit
def scan_op(l, r):
    xl, fl = unpack64(l)
    xr, fr = unpack64(r)
    x = xl * fr + xr
    f = fl * fr
    return pack64(x, f)

@triton.jit
def scan3(
    gates,
    tokens,
    outputs,
    SEQUENCE_LENGTH: tl.constexpr,
):
    global_id = tl.num_programs(axis=1) * tl.program_id(axis=0) + tl.program_id(axis=1)
    offsets = tl.arange(0, SEQUENCE_LENGTH) + global_id * SEQUENCE_LENGTH

    tokens_ = tl.load(tokens + offsets)
    gates_ = tl.load(gates + offsets)

    tuples = pack64(tokens_, gates_)
    output_tuples = tl.associative_scan(tuples, 0, combine_fn=scan_op)
    output_tokens, output_gates = unpack64(output_tuples)
    tl.store(outputs + offsets, output_tokens)

@triton.jit(debug=True)
def scan4(
    gates,
    tokens,
    outputs,
    ports,
    SEQUENCE_LENGTH: tl.constexpr,
    CHUNK_LENGTH: tl.constexpr,
):
    global_id = tl.num_programs(axis=1) * tl.program_id(axis=0) + tl.program_id(axis=1)
    seq_offset = global_id * SEQUENCE_LENGTH
    chunk = tl.program_id(axis=2) * CHUNK_LENGTH

    out = 0.
    port = 1.
    for unroll_i in tl.static_range(CHUNK_LENGTH):
        i = chunk + unroll_i
        gate = tl.load(gates + seq_offset + i)
        port = port * gate
        if i == 0:
            out = tl.load(tokens + seq_offset + i)
        else:
            out = gate * out + tl.load(tokens + seq_offset + i)
        tl.store(outputs + seq_offset + i, out)
        tl.store(ports + seq_offset + i, port)

    tl.debug_barrier()
    if chunk == 0:
        tl.debug_barrier()

        # stitch all chunks
        indices = tl.arange(0, CHUNK_LENGTH)
        last_index = CHUNK_LENGTH - 1
        last_out = tl.load(outputs + seq_offset + last_index)

        for chunk_i in range(1, tl.num_programs(axis=2)):
            offsets = seq_offset + chunk_i * CHUNK_LENGTH + indices
            chunk_outputs = tl.load(outputs + offsets)
            chunk_ports = tl.load(ports + offsets)
            chunk_outputs = chunk_ports * last_out + chunk_outputs
            tl.store(outputs + offsets, chunk_outputs)
            last_out = tl.load(outputs + seq_offset + chunk_i * CHUNK_LENGTH + last_index)
            ## WHY DO YOU WORK ONLY WITH THIS PRINT?
            # tl.device_print(
            #     "stitch ", tl.program_id(axis=0), tl.program_id(axis=1), tl.program_id(axis=2), chunk_i,
            # )

class Scan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gates, tokens):
        B, C, T = gates.shape
        assert tokens.shape == (B, C, T)

        states = torch.zeros_like(tokens)
        scan2[(B,C)](gates, tokens, states, T, UNROLL_LENGTH=64)

        ctx.save_for_backward(states, gates)
        return states

    # backward scan is a padded reverse scan
    # https://arxiv.org/abs/1709.04057 Section 2.2
    @staticmethod
    def backward(ctx, grad_output):
        states, gates = ctx.saved_tensors
        B, C, T = gates.shape

        grad_output = grad_output.contiguous()
        assert states.is_contiguous()
        assert gates.is_contiguous()

        d_states = torch.zeros_like(states)
        scan2_backward[(B,C)](gates, grad_output.contiguous(), d_states, T, UNROLL_LENGTH=64)

        outputs_pad = torch.cat([torch.zeros_like(states[:, :, :1]), states], dim=-1)[:, :, :-1]
        d_gates = outputs_pad * d_states

        d_tokens = d_states
        return d_gates, d_tokens

@triton.testing.perf_report([
    triton.testing.Benchmark(
        x_names=["SEQUENCE_LENGTH"],  # argument names to use as an x-axis for the plot
        x_vals=[2**i for i in range(17)],
        line_arg="provider",  # argument name whose value corresponds to a different line in the plot
        #line_vals=["scan1", "scan2", "tl.associative_scan", "eager"],  # argument values to use as different lines in the plot
        #line_names=["scan1", "scan2", "tl.associative_scan", "eager"],  # legend entries to use for each line
        line_vals=["scan1", "scan2", "tl.associative_scan"],
        line_names=["scan1", "scan2", "tl.associative_scan"],
        plot_name="scan1",  # name of the plot
        args={
            #"SEQUENCE_LENGTH": seq_len,
        }
    ),
    triton.testing.Benchmark(
        x_names=["SEQUENCE_LENGTH"],
        x_vals=[2**i for i in range(17)],
        line_arg="UNROLL_LENGTH",
        line_vals=[1,2,4,8,16,32,64,128,256],
        line_names=[str(s) for s in [1,2,4,8,16,32,64,128,256]],
        plot_name="scan2",
        args={
            "provider": "scan2",
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
def bench(provider, SEQUENCE_LENGTH, UNROLL_LENGTH=64, device="cuda"):
    B, C, T = 1, 1, SEQUENCE_LENGTH
    gates, tokens = init(B, C, T, device)
    outputs = torch.empty_like(tokens)

    match provider:
        case "scan1":
            scan = lambda: scan1[(1,)](gates, tokens, outputs, SEQUENCE_LENGTH)
        case "scan2":
            scan = lambda: scan2[(1,)](gates, tokens, outputs, SEQUENCE_LENGTH, UNROLL_LENGTH)
        case "tl.associative_scan":
            scan = lambda: scan3[(1,)](gates, tokens, outputs, SEQUENCE_LENGTH)
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
            outputs[:, :, i] = gates[:, :, i] * outputs[:, :, i-1].clone() + tokens[:, :, i]

def test_allclose():
    device = 'cuda'
    B, C, T = 1, 1, 512
    gates, tokens = init(B, C, T, device)
    outputs = torch.empty_like(tokens)
    scan1[(1,)](gates, tokens, outputs, T)

    outputs2 = torch.empty_like(tokens)
    scan2[(1,)](gates, tokens, outputs2, T, UNROLL_LENGTH=16)

    assert torch.allclose(outputs, outputs2)

    outputs_naive = torch.empty_like(tokens)
    scan_eager(gates, tokens, outputs_naive)

    assert torch.allclose(outputs, outputs_naive)

def test_allclose2():
    device = 'cuda'
    B, C, T = 8, 16, 512
    gates, tokens = init(B, C, T, device)

    outputs = torch.empty_like(tokens)
    scan2[(B,C)](gates, tokens, outputs, T, UNROLL_LENGTH=16)

    outputs_naive = torch.empty_like(tokens)
    scan_eager(gates, tokens, outputs_naive)

    assert torch.allclose(outputs, outputs_naive)

def test_allclose3():
    device = 'cuda'
    B, C, T = 1,1, 512
    gates, tokens = init(B, C, T, device)

    outputs = torch.empty_like(tokens)
    scan2[(B,C)](gates, tokens, outputs, T, UNROLL_LENGTH=16)

    outputs3 = torch.empty_like(tokens)
    scan3[(B,C)](gates, tokens, outputs3, T)
    print(outputs)
    print(outputs3)
    assert torch.allclose(outputs, outputs3, atol=1e-1)

def test_backward():
    device = 'cuda'
    B, C, T = 4, 8, 512
    gates, tokens = init(B, C, T, device)
    gates.requires_grad = True
    tokens.requires_grad = True

    outputs_eager = torch.empty_like(tokens)
    scan_eager(gates, tokens, outputs_eager)
    outputs_eager.sum().backward()

    gates_auto_grad = gates.grad.clone()
    tokens_auto_grad = tokens.grad.clone()
    gates.grad = None
    tokens.grad = None

    outputs = Scan.apply(gates, tokens)
    outputs.sum().backward()

    # print(outputs)
    # print(outputs_eager)
    assert torch.allclose(outputs, outputs_eager)

    print(tokens_auto_grad[0,0,:64])
    print(tokens.grad[0,0,:64])
    assert torch.allclose(tokens_auto_grad, tokens.grad)

    print(gates_auto_grad[0,0,:64])
    print(gates.grad[0,0,:64])
    assert torch.allclose(gates_auto_grad, gates.grad)


def test_grid():
    device = 'cuda'
    B, C, T = 4, 4, 4
    CHUNK_LENGTH = 2
    gates, tokens = init(B, C, T, device)

    outputs_eager = torch.empty_like(tokens)
    scan_eager(gates, tokens, outputs_eager)

    outputs = torch.empty_like(tokens)
    ports = torch.empty_like(gates)
    scan4[(B,C,T//CHUNK_LENGTH)](gates, tokens, outputs, ports, T, CHUNK_LENGTH=CHUNK_LENGTH, num_warps=1)
    torch.cuda.synchronize()
    print()
    print(outputs - outputs_eager)
    assert torch.allclose(outputs, outputs_eager)

if __name__ == '__main__':
    bench.run(save_path=".", print_data=True)