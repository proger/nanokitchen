# https://nvidia.github.io/cuda-python/overview.html#cuda-python-workflow

from cuda import cuda, cudart

err, = cuda.cuInit(0)
err, deviceCount = cuda.cuDeviceGetCount()

def print_level(title, keys, properties, depth=0):
    print()
    print(" "*(2*depth) + title + ":")

    if keys is None:
        keys = properties.keys()
    
    for key in keys:
        value = properties.get(key) if isinstance(properties, dict) else getattr(properties, key)
        print(" "*(2*depth+1) + key, value)

print("See https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html for detailed descriptions of parameters")
for dev in range(deviceCount):
    error, driverVersion = cuda.cuDriverGetVersion()
    error, properties = cudart.cudaGetDeviceProperties(dev)

    err, cuDevice = cuda.cuDeviceGet(0)
    err, context = cuda.cuCtxCreate(0, cuDevice)
    err, free, total = cuda.cuMemGetInfo()
    err, = cuda.cuCtxDestroy(context)

    print_level("Device", [
        "id",
        "name",
        "driver",
        "SM"
    ], {
        "id": dev,
        "name": properties.name.decode("utf-8"),
        "driver": driverVersion,
        "SM": f"{properties.major}.{properties.minor}",
    }, depth=0)

    print_level("Grid Parameters", None, {
        "freeGlobalMem": free,
        "totalGlobalMem": total,
        "memoryClockRate": properties.memoryClockRate,
        "memoryBusWidth": properties.memoryBusWidth,
        #"memPitch": properties.memPitch,
        "totalConstMem": properties.totalConstMem,
        "l2CacheSize": properties.l2CacheSize,
        "persistingL2CacheMaxSize": properties.persistingL2CacheMaxSize,
        "multiProcessorCount": properties.multiProcessorCount,
    }, depth=1)

    print_level("SM Parameters", [
        "regsPerMultiprocessor",
        "maxThreadsPerMultiProcessor",
        "sharedMemPerMultiprocessor",
        "maxBlocksPerMultiProcessor",
    ], properties, depth=2)

    print_level("Block Parameters", [
        "maxThreadsDim",
        "sharedMemPerBlock",
        "maxThreadsPerBlock",
        "regsPerBlock",
        "reservedSharedMemPerBlock",
    ], properties, depth=3)

    print_level("Warp Parameters", [
        "warpSize",
    ], properties, depth=4)

    print_level("Thread Parameters", [
        "clockRate",
    ], properties, depth=5)