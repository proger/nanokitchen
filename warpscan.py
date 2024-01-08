from numba import cuda
import numpy as np


def make_warp_scan(
    STEPS_PER_THREAD, # sequential steps in the beginning of the algorithm
    WARPS_PER_BLOCK
):
    MAX_THREADS_PER_BLOCK = 1024
    assert MAX_THREADS_PER_BLOCK == 1024
    WARP_SIZE = 32
    assert WARP_SIZE == 32
    LOG_WARP_SIZE = 5 # log2(threads_per_warp)

    block_dim = WARP_SIZE * WARPS_PER_BLOCK
    assert block_dim <= MAX_THREADS_PER_BLOCK
    N = block_dim * STEPS_PER_THREAD

    @cuda.jit(fastmath=False, lineinfo=True)
    def scan(gates, tokens, result):
        def mempty():
            return np.float32(1.), np.float32(0.)
        def mappend(fl, xl, fr, xr):
            return fl * fr, xl * fr + xr

        thread_id = cuda.threadIdx.x
        warp_id = thread_id // WARP_SIZE
        lane_id = thread_id % WARP_SIZE

        #
        # Perform a little bit of sequential computation in thread registers.
        #

        thread_acc_f = cuda.local.array(STEPS_PER_THREAD, np.float32)
        thread_acc_x = cuda.local.array(STEPS_PER_THREAD, np.float32)

        acc_f, acc_x = mempty()
        for i in range(0, STEPS_PER_THREAD):
            g = np.float32(gates[thread_id * STEPS_PER_THREAD + i])
            t = np.float32(tokens[thread_id * STEPS_PER_THREAD + i])
            acc_f, acc_x = mappend(acc_f,
                                   acc_x,
                                   g,
                                   t)
            thread_acc_f[i], thread_acc_x[i] = acc_f, acc_x

        cuda.syncthreads()

        #
        # Stitch threads in a warp using shuffling.
        #

        for e in range(0, LOG_WARP_SIZE):
            delta = 1 << e

            # At the same time:
            # Send acc to the thread with id (lane_id + delta)
            # Receive acc of the thread with id (lane_id - delta)
            recv_f = cuda.shfl_up_sync(0xffffffff, acc_f, delta)
            recv_x = cuda.shfl_up_sync(0xffffffff, acc_x, delta)

            if lane_id >= delta:
                for i in range(0, STEPS_PER_THREAD):
                    acc_f, acc_x = mappend(thread_acc_f[i], thread_acc_x[i], recv_f, recv_x)
                    thread_acc_f[i], thread_acc_x[i] = acc_f, acc_x

        cuda.syncthreads()

        #
        # Stitch warps in a block using shared memory.
        #

        warp_last_f = cuda.shared.array(shape=WARPS_PER_BLOCK, dtype=np.float32)
        warp_last_x = cuda.shared.array(shape=WARPS_PER_BLOCK, dtype=np.float32)
        if lane_id == WARP_SIZE - 1:
            warp_last_f[warp_id] = acc_f
            warp_last_x[warp_id] = acc_x

        cuda.syncthreads()

        if WARPS_PER_BLOCK <= 4:
            # if there are at most 4 warps per block, do a sequential scan over shared memory
            if thread_id == 0:
                warp_acc_f, warp_acc_x = mempty()
                for w in range(0, WARPS_PER_BLOCK):
                    warp_acc_f, warp_acc_x = mappend(warp_acc_f,
                                                     warp_acc_x,
                                                     warp_last_f[w],
                                                     warp_last_x[w])
                    warp_last_f[w] = warp_acc_f
                    warp_last_x[w] = warp_acc_x
        else:
            # otherwise do a warp scan on warp 0
            if warp_id == 0:
                if lane_id < WARPS_PER_BLOCK:
                    warp_acc_f = warp_last_f[lane_id]
                    warp_acc_x = warp_last_x[lane_id]
                else:
                    warp_acc_f, warp_acc_x = mempty()

                for e in range(0, LOG_WARP_SIZE):
                    delta = 1 << e
                    recv_f = cuda.shfl_up_sync(0xffffffff, warp_acc_f, delta)
                    recv_x = cuda.shfl_up_sync(0xffffffff, warp_acc_x, delta)

                    if lane_id >= delta:
                        warp_acc_f, warp_acc_x = mappend(warp_acc_f, warp_acc_x, recv_f, recv_x)

                if lane_id < WARPS_PER_BLOCK:
                    warp_last_f[lane_id] = warp_acc_f
                    warp_last_x[lane_id] = warp_acc_x
                    #print('warp', warp_id, 'lane', lane_id, 'warp_acc', warp_acc)

        cuda.syncthreads()

        # Add the last element of the previous warp to each element of the current warp.
        if warp_id > 0:
            warp_from_left_f = warp_last_f[warp_id - 1]
            warp_from_left_x = warp_last_x[warp_id - 1]
        else:
            warp_from_left_f, warp_from_left_x = mempty()

        for i in range(0, STEPS_PER_THREAD):
            thread_acc_f[i], thread_acc_x[i] = mappend(thread_acc_f[i],
                                                       thread_acc_x[i],
                                                       warp_from_left_f,
                                                       warp_from_left_x)
            result[thread_id * STEPS_PER_THREAD + i] = thread_acc_x[i]

        #print('thread', thread_id, 'warp', warp_id, 'lane', lane_id, 'acc', i, thread_acc[i])

    return scan, block_dim, N
