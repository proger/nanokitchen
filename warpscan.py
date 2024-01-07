from numba import cuda
import numpy as np

MAX_THREADS_PER_BLOCK = 1024
assert MAX_THREADS_PER_BLOCK == 1024
WARP_SIZE = 32
assert WARP_SIZE == 32
LOG_WARP_SIZE = 5 # log2(threads_per_warp)

STEPS_PER_THREAD = 4 # sequential steps in the beginning of the algorithm
WARPS_PER_BLOCK = 4
block_dim = WARP_SIZE * WARPS_PER_BLOCK
assert block_dim <= MAX_THREADS_PER_BLOCK
N = block_dim * STEPS_PER_THREAD

@cuda.jit
def scan(data, result):
    def mzero():
        return 0
    def mappend(a, b):
        return a + b

    thread_id = cuda.threadIdx.x
    warp_id = thread_id // WARP_SIZE
    lane_id = thread_id % WARP_SIZE

    #
    # Perform a little bit of sequential computation in thread registers.
    #

    thread_acc = cuda.local.array(STEPS_PER_THREAD, np.float32)

    acc = mzero()
    for i in range(0, STEPS_PER_THREAD):
        acc = mappend(acc, data[thread_id * STEPS_PER_THREAD + i])
        thread_acc[i] = acc

    cuda.syncthreads()

    #
    # Stitch threads in a warp using shuffling.
    #

    for e in range(0, LOG_WARP_SIZE):
        delta = 1 << e

        # At the same time:
        # Send acc to the thread with id (lane_id + delta)
        # Receive acc of the thread with id (lane_id - delta)
        recv = cuda.shfl_up_sync(0xffffffff, acc, delta)

        if lane_id >= delta:
            for i in range(0, STEPS_PER_THREAD):
                acc = mappend(thread_acc[i], recv)
                thread_acc[i] = acc

    cuda.syncthreads()

    #
    # Stitch warps in a block using shared memory.
    #

    warp_last = cuda.shared.array(shape=WARPS_PER_BLOCK, dtype=np.float32)
    if lane_id == WARP_SIZE - 1:
        warp_last[warp_id] = acc

    cuda.syncthreads()

    if WARPS_PER_BLOCK <= 4:
        # if there are at most 4 warps per block, do a sequential scan over shared memory
        if thread_id == 0:
            warp_acc = mzero()
            for w in range(0, WARPS_PER_BLOCK):
                warp_acc = mappend(warp_acc, warp_last[w])
                warp_last[w] = warp_acc
    else:
        # otherwise do a warp scan on warp 0
        if warp_id == 0:
            if lane_id < WARPS_PER_BLOCK:
                warp_acc = warp_last[lane_id]
            else:
                warp_acc = mzero()

            for e in range(0, LOG_WARP_SIZE):
                delta = 1 << e
                recv = cuda.shfl_up_sync(0xffffffff, warp_acc, delta)

                if lane_id >= delta:
                    warp_acc = mappend(warp_acc, recv)

            if lane_id < WARPS_PER_BLOCK:
                warp_last[lane_id] = warp_acc
                #print('warp', warp_id, 'lane', lane_id, 'warp_acc', warp_acc)

    cuda.syncthreads()

    # Add the last element of the previous warp to each element of the current warp.
    if warp_id > 0:
        warp_from_left = warp_last[warp_id - 1]
    else:
        warp_from_left = 0

    for i in range(0, STEPS_PER_THREAD):
        thread_acc[i] = mappend(thread_acc[i], warp_from_left)
        result[thread_id * STEPS_PER_THREAD + i] = thread_acc[i]

    #print('thread', thread_id, 'warp', warp_id, 'lane', lane_id, 'acc', i, thread_acc[i])


# approximation for your mental model
def scan_sim(data, WARP_SIZE=32):
    STEPS_PER_THREAD = 1 # not simulating these
    WARPS_PER_BLOCK = data.shape[0] // WARP_SIZE

    #
    # Stitch threads in a warp using shuffling.
    #

    result = np.zeros_like(data)
    for w in range(WARPS_PER_BLOCK):
        interval = slice(w * WARP_SIZE, (w + 1) * WARP_SIZE)
        result[interval] = np.cumsum(data[interval])

    #
    # Stitch warps in a block using shared memory.
    #

    warp_last = np.zeros(WARPS_PER_BLOCK)
    for w in range(1, WARPS_PER_BLOCK):
        warp_last[w] = warp_last[w-1] +  result[(w-1) * WARP_SIZE + WARP_SIZE - 1]
    rep = np.repeat(warp_last, WARP_SIZE)

    result = result + rep
    
    return result


if __name__ == '__main__':
    data = np.arange(N).astype(np.float32)

    result = np.zeros_like(data)
    scan[(1,), block_dim](data, result)

    print(data, "data")
    print(result, "result")
    ref = np.cumsum(data)
    print(ref, "cumsum")
    assert np.allclose(result, ref)
    sim = scan_sim(data)
    #print(sim, "ref")
    assert np.allclose(sim, ref)
    print(N)