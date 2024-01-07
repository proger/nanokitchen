from numba import cuda
import numpy as np

threads_per_warp = 32
log_threads_per_warp = 5 # log2(threads_per_warp)
warps_per_block = 1
N = threads_per_warp * warps_per_block

@cuda.jit
def scan(data, result):
    def mappend(a, b):
        return a + b

    thread_id = cuda.threadIdx.x
    warp_id = thread_id // threads_per_warp
    lane_id = thread_id % threads_per_warp

    # Copy data from global memory into the register file of each thread.
    acc = data[thread_id]

    cuda.syncthreads()

    for e in range(0, log_threads_per_warp):
        delta = 1 << e

        # At the same time:
        # Send acc to the thread with id (lane_id + delta)
        # Receive acc of the thread with id (lane_id - delta)
        recv = cuda.shfl_up_sync(0xffffffff, acc, delta)

        temp = mappend(recv, acc)
        if lane_id >= delta:
            acc = temp

    result[thread_id] = acc
    #print('thread', thread_id, 'warp', warp_id, 'lane', lane_id, 'acc', acc)


if __name__ == '__main__':
    data = np.arange(N).astype(np.float32)

    result = np.zeros_like(data)
    scan[(1,), (N,)](data, result)

    print(data, "data")
    print(result, "result")
    print(np.cumsum(data), "cumsum")