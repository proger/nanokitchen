import math
import torch
import torch.nn.functional as F
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

# fast but inaccurate, see test_allclose3
@triton.jit
def scan3(
    gates,
    tokens,
    outputs,
    output_gates,
    SEQUENCE_LENGTH: tl.constexpr,
):
    global_id = tl.num_programs(axis=1) * tl.program_id(axis=0) + tl.program_id(axis=1)
    offsets = tl.arange(0, SEQUENCE_LENGTH) + global_id * SEQUENCE_LENGTH

    tokens_ = tl.load(tokens + offsets)
    gates_ = tl.load(gates + offsets)

    tuples = pack64(tokens_, gates_)
    output_tuples = tl.associative_scan(tuples, axis=0, combine_fn=scan_op)
    output_tokens, output_gates1 = unpack64(output_tuples)
    # when I exchange these two stores outputs blow up! why?
    tl.store(output_gates + offsets, output_gates1)
    tl.store(outputs + offsets, output_tokens)

@triton.jit()
def scan4(
    gates,
    tokens,
    outputs,
    ports,
    sequence_locks,
    SEQUENCE_LENGTH: tl.constexpr,
    CHUNK_LENGTH: tl.constexpr,
):
    sequence_id = tl.num_programs(axis=1) * tl.program_id(axis=0) + tl.program_id(axis=1)
    seq_stride = sequence_id * SEQUENCE_LENGTH
    other_chunks = tl.num_programs(axis=2)-1
    chunk = tl.program_id(axis=2) * CHUNK_LENGTH
    Lock = sequence_locks + sequence_id

    out = 0.
    port = 1.
    for unroll_i in range(CHUNK_LENGTH):
        i = chunk + unroll_i
        gate = tl.load(gates + seq_stride + i)
        port = port * gate
        if i == 0:
            out = tl.load(tokens + seq_stride + i)
        else:
            out = gate * out + tl.load(tokens + seq_stride + i)

        tl.store(outputs + seq_stride + i, out)
        tl.store(ports + seq_stride + i, port)

    if chunk == 0:
        # wait for all chunks to complete
        while tl.atomic_min(Lock, other_chunks) != other_chunks:
            pass

        # stitch all chunks
        indices = tl.arange(0, CHUNK_LENGTH)
        last_index = CHUNK_LENGTH - 1
        last_out = tl.load(outputs + seq_stride + last_index)

        for chunk_i in range(1, tl.num_programs(axis=2)):
            strides = seq_stride + chunk_i * CHUNK_LENGTH + indices
            chunk_outputs = tl.load(outputs + strides)
            chunk_ports = tl.load(ports + strides)
            chunk_outputs1 = chunk_ports * last_out + chunk_outputs
            tl.store(outputs + strides, chunk_outputs1)
            last_out = tl.load(outputs + seq_stride + chunk_i * CHUNK_LENGTH + last_index)

            # tl.device_assert(chunk_ports != 42, "CANARY DETECTED: MISSING WRITE TO ports")  # i messed up locking the first time
            # if tl.program_id(axis=0) == 0 and tl.program_id(axis=1) == 2:
            #     tl.device_print(
            #         "stitch ", tl.program_id(axis=0), tl.program_id(axis=1), tl.program_id(axis=2), chunk_i, chunk_outputs, last_out, chunk_ports, indices
            #     )
    else:
        tl.atomic_add(Lock, 1)

# https://github.com/glassroom/heinsen_sequence
@torch.compile
def heinsen_positive(log_coeffs, log_values):
    a_star = torch.cumsum(log_coeffs, dim=-1)                             # eq (2) in paper
    log_x0_plus_b_star = torch.logcumsumexp(log_values - a_star, dim=-1)  # eq (7) in paper
    log_x = a_star + log_x0_plus_b_star                                   # eq (1) in paper
    return torch.exp(log_x)                                               # already a float

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


def split(x):
  """
  Take a sequence of inputs that represents a tree level,
  and return all left children and all right children.

  >>> split(torch.tensor([1,2,3,4,5,6,7,8])[None, None, :])
  (tensor([[[1, 3, 5, 7]]]), tensor([[[2, 4, 6, 8]]])
  """
  B, C, T = x.size()
  x = x.view(B, C, T//2, 2)
  return x[: , :, :, 0], x[:, :, :, 1]

def merge(lefts, rights):
  """
  Take sequences of all left children and sequences of all right children and merge them
  into a single tree level.

  >>> lefts = torch.tensor([1,3,5,7])[None, None, :]
  >>> rights = torch.tensor([2,4,6,8])[None, None, :]
  >>> merge(lefts, rights)
  tensor([[[1, 2, 3, 4, 5, 6, 7, 8]]])
  """
  B, C, half = lefts.size()
  x = torch.stack([lefts, rights], dim=-1) # (bsz, dim, half, 2)
  return x.view(B, C, half*2)

@torch.compile
def parallel_scan(gates, x, mul=torch.mul, add=torch.add, zeros_like=torch.zeros_like):
  B,C,T = x.size()
  level = int(math.log2(T))
  return add(mul(parallel_scan1(gates, x, mul, add, zeros_like, level=level), gates), x)

def parallel_scan1(gates, x, mul, add, zeros_like, level):
  left_gates, right_gates = split(gates)
  left_x, right_x = split(x)

  # up: sum together
  gates = mul(left_gates, right_gates)
  x = add(mul(right_gates, left_x), right_x)

  if level == 1:
      root_x = zeros_like(x)
  else:
      root_x = parallel_scan1(gates, x, mul, add, zeros_like, level=level-1)

  # down: push from left to right
  return merge(root_x, add(mul(root_x, left_gates), left_x))


@triton.testing.perf_report([
    triton.testing.Benchmark(
        x_names=["SEQUENCE_LENGTH"],  # argument names to use as an x-axis for the plot
        x_vals=[2**i for i in range(7,17)],
        xlabel='sequence length',
        ylabel='ms',
        x_log=True,
        y_log=True,
        line_arg="provider",  # argument name whose value corresponds to a different line in the plot
        #line_vals=["scan1", "scan2", "tl.associative_scan", "eager"],  # argument values to use as different lines in the plot
        #line_names=["scan1", "scan2", "tl.associative_scan", "eager"],  # legend entries to use for each line
        line_vals=["scan1", "scan2", "tl.associative_scan", "scan4", "parallel_scan", "heinsen_positive", "cub"],
        line_names=["scan1", "scan2", "tl.associative_scan (fast but wrong)", "scan4", "parallel_scan (with torch.compile)", "Heinsen", "cub"],
        plot_name="scan1",  # name of the plot
        args={
            #"SEQUENCE_LENGTH": seq_len,
        }
    ),
    # triton.testing.Benchmark(
    #     x_names=["SEQUENCE_LENGTH"],
    #     x_vals=[2**i for i in range(7,17)],
    #     line_arg="CHUNK_LENGTH",
    #     line_vals=[1,2,4,8,16,32,64,128,256],
    #     line_names=[str(s) for s in [1,2,4,8,16,32,64,128,256]],
    #     plot_name="scan2",
    #     args={
    #         "provider": "scan2",
    #     }
    # ),
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
def bench(provider, SEQUENCE_LENGTH, CHUNK_LENGTH=64, device="cuda"):
    B, C, T = 1, 1024, SEQUENCE_LENGTH
    gates, tokens = init(B, C, T, device)
    outputs = torch.empty_like(tokens)

    match provider:
        case "scan1":
            print(f"Running {provider} with sequence length {SEQUENCE_LENGTH}")
            scan = lambda: scan1[(B,C)](gates, tokens, outputs, SEQUENCE_LENGTH)
        case "scan2":
            print(f"Running {provider} with sequence length {SEQUENCE_LENGTH} and chunk length {CHUNK_LENGTH}")
            scan = lambda: scan2[(B,C)](gates, tokens, outputs, SEQUENCE_LENGTH, CHUNK_LENGTH)
        case "tl.associative_scan":
            print(f"Running {provider} with sequence length {SEQUENCE_LENGTH}")
            output_gates = torch.zeros_like(gates).contiguous()
            scan = lambda: scan3[(B,C)](gates, tokens, outputs, output_gates, SEQUENCE_LENGTH)
        case "scan4":
            print(f"Running {provider} with sequence length {SEQUENCE_LENGTH} and chunk length {CHUNK_LENGTH}")
            outputs = torch.zeros_like(tokens).contiguous()
            ports = torch.zeros_like(gates).contiguous() + 42
            locks = torch.zeros((B, C), dtype=torch.int32, device=device).contiguous()
            if SEQUENCE_LENGTH < CHUNK_LENGTH:
                scan = lambda: scan4[(B,C,1)](gates, tokens, outputs, ports, locks,
                                              SEQUENCE_LENGTH=SEQUENCE_LENGTH,
                                              CHUNK_LENGTH=SEQUENCE_LENGTH,
                                              num_warps=1)
            else:
                c = int(math.sqrt(SEQUENCE_LENGTH))
                CHUNK_LENGTH = triton.next_power_of_2(triton.cdiv(SEQUENCE_LENGTH, c))
                print("scan4: grid axis z has size", T//CHUNK_LENGTH)
                scan = lambda: scan4[(B,C,T//CHUNK_LENGTH)](gates, tokens, outputs, ports, locks,
                                                            SEQUENCE_LENGTH=SEQUENCE_LENGTH,
                                                            CHUNK_LENGTH=CHUNK_LENGTH,
                                                            num_warps=1)
        case "parallel_scan":
            print(f"Running {provider} with sequence length {SEQUENCE_LENGTH}")
            scan = lambda: parallel_scan(gates, tokens)
        case "heinsen_positive":
            print(f"Running {provider} with sequence length {SEQUENCE_LENGTH}")
            scan = lambda: heinsen_positive(gates.abs().log(), tokens.abs().log())
        case "cub":
            print(f"Running {provider} with sequence length {SEQUENCE_LENGTH}")
            from cuscan import simple_scan_forward
            scan = lambda: simple_scan_forward(gates, tokens)
        case "eager":
            print(f"Running {provider} with sequence length {SEQUENCE_LENGTH}")
            scan = lambda: scan_eager(gates, tokens, outputs)
        case "compile":
            print(f"Running {provider} with sequence length {SEQUENCE_LENGTH}")
            # compilation times are proportional to sequence length!
            compiled = torch.compile(scan_eager)
            scan = lambda: compiled(gates, tokens, outputs)
        case _:
            raise ValueError(f"Unknown provider {provider}")

    # large warmup for benefit of torch.compile
    ms = triton.testing.do_bench(scan, warmup=1000, rep=100)
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
    B, C, T = 1, 512, 64
    gates, tokens = init(B, C, T, device)

    outputs = torch.empty_like(tokens)
    #scan2[(B,C)](gates, tokens, outputs, T, UNROLL_LENGTH=16)
    scan_eager(gates, tokens, outputs)

    outputs3 = torch.zeros_like(tokens).contiguous()
    output_gates = torch.zeros_like(gates).contiguous()
    scan3[(B,C)](gates, tokens, outputs3, output_gates, T)

    print('max gate error', (output_gates - gates.cumprod(dim=-1)).abs().max())
    #print(outputs)
    #print(outputs3)
    print('max error', (outputs - outputs3).abs().max())
    # LOOK AT THIS MASSIVE atol:
    assert torch.allclose(outputs, outputs3, atol=5e-1)

def test_backward():
    device = 'cuda'
    B, C, T = 1, 8, 512
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
    B, C, T = 1, 1024, 128
    CHUNK_LENGTH = 64
    torch.manual_seed(12312323)
    gates, tokens = init(B, C, T, device)

    # gates, tokens = init(B, 1, T, device)
    # gates = gates.repeat(1,C,1).clone().contiguous()
    # tokens = tokens.repeat(1,C,1).clone().contiguous()

    outputs_eager = torch.empty_like(tokens)
    scan_eager(gates, tokens, outputs_eager)

    outputs = torch.zeros_like(tokens).contiguous()
    ports = torch.zeros_like(gates).contiguous() + 42
    locks = torch.zeros((B,C), dtype=torch.int32, device=device).contiguous()
    scan4[(B,C,T//CHUNK_LENGTH)](gates, tokens, outputs, ports, locks, SEQUENCE_LENGTH=T, CHUNK_LENGTH=CHUNK_LENGTH, num_warps=1)
    print()
    print(locks, 'locks')
    print(torch.where(~torch.isclose(outputs, outputs_eager)))
    print(outputs)
    print(outputs_eager[0,-1,:])
    assert torch.allclose(outputs, outputs_eager)

def test_heinsen():
    device = 'cuda'
    B, C, T = 1, 1024, 128
    torch.manual_seed(12312323)
    gates, tokens = init(B, C, T, device)
    gates = gates.abs()
    tokens = tokens.abs()

    outputs_eager = torch.empty_like(tokens)
    scan_eager(gates, tokens, outputs_eager)

    outputs = heinsen_positive(gates.log(), tokens.log())
    assert torch.allclose(outputs, outputs_eager)

def test_allclose_cub():
    device = 'cuda'
    B, C, T = 1, 1024, 128
    torch.manual_seed(12312323)
    gates, tokens = init(B, C, T, device)

    outputs_eager = torch.empty_like(tokens)
    scan_eager(gates, tokens, outputs_eager)

    from cuscan import simple_scan_forward
    outputs, _ = simple_scan_forward(gates, tokens)
    assert torch.allclose(outputs, outputs_eager)

if __name__ == '__main__':
    bench.run(save_path=".", print_data=True)