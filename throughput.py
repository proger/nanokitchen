import time
import torch
import torch.nn
import torch.nn.functional as F

torch.set_float32_matmul_precision('high')

device = 'cuda'
vocab_size = 16384
scan_batch_size = False
test_throughput = True
#target_tokens = 200_000_000_000
target_tokens = 21_400_000_000
arch = 'mamba'

match arch:
  case 'mamba':
    from mamba_ssm import Mamba, MambaLMHeadModel
    from mamba_ssm.models.config_mamba import MambaConfig

    #batch_size, seq_len = 4, 8192
    #batch_size, seq_len = 14, 512
    batch_size, seq_len = 1, 4096
    config = MambaConfig(
      d_model=1536,
      n_layer=48,
      vocab_size=vocab_size,
      ssm_cfg={},
      rms_norm=True,
      fused_add_norm=True,
      residual_in_fp32=True,
    )

    model = MambaLMHeadModel(
      config=config
    )
    model = model.to(device)

    def forward_backward(inputs, targets):
      with torch.autocast('cuda', dtype=torch.float16):
        output = model(inputs)
        y = output.logits
        loss = F.cross_entropy(y.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        return loss

  case 'gpt':
    batch_size, seq_len = 2, 1024
    from model import GPT, GPTConfig 
    config = GPTConfig(block_size=seq_len)
    model = GPT(config).to(device)
    model = torch.compile(model)

    def forward_backward(inputs, targets):
      with torch.autocast('cuda', dtype=torch.bfloat16):
        logits, loss = model(inputs, targets=targets)
        loss.backward()
        return loss

print(sum(p.numel() for p in model.parameters()))

if scan_batch_size:
  for test_batch_size in range(2, batch_size*2):
    x = torch.randint(vocab_size,(test_batch_size,seq_len)).to(device)
    y = torch.randint(vocab_size,(test_batch_size,seq_len)).to(device)
    try:
      model.zero_grad(set_to_none=True)
      loss = forward_backward(x, y)
    except torch.cuda.OutOfMemoryError:
      print('OOM') 
      break
    else:
      batch_size = test_batch_size
      print('batch_size = ', batch_size)

if test_throughput:
  x = torch.randint(vocab_size,(batch_size,seq_len)).to(device)
  y = torch.randint(vocab_size,(batch_size,seq_len)).to(device)
  print('warmup')
  for _ in range(4):
    model.zero_grad(set_to_none=True)
    loss = forward_backward(x, y)
  total_tokens = 0
  t0 = time.perf_counter()
  test_batches = 32
  for _ in range(test_batches):
    model.zero_grad(set_to_none=True)
    loss = forward_backward(x, y)
    total_tokens += x.numel()
    print(total_tokens)
  t1 = time.perf_counter()
  dt = t1-t0

  tps = total_tokens / dt
  total_seconds = target_tokens / tps
  total_hours = total_seconds / 3600
  total_days = total_hours / 24
  print('tps', tps, 'total days', round(total_days, 2), 'or hours', round(total_hours, 2))
