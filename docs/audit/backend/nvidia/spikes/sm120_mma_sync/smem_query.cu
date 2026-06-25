#include <cstdio>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e=(x); if(e){printf("CUDA error %s at %d: %s\n",#x,__LINE__,cudaGetErrorString(e));return 1;} } while(0)

int main() {
    int dev = 0;
    CK(cudaSetDevice(dev));
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, dev));

    int optin = 0, perSM = 0, perBlock = 0, reservedPerBlock = 0;
    CK(cudaDeviceGetAttribute(&optin,            cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    CK(cudaDeviceGetAttribute(&perSM,            cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev));
    CK(cudaDeviceGetAttribute(&perBlock,         cudaDevAttrMaxSharedMemoryPerBlock, dev));
#ifdef cudaDevAttrReservedSharedMemoryPerBlock
    CK(cudaDeviceGetAttribute(&reservedPerBlock, cudaDevAttrReservedSharedMemoryPerBlock, dev));
#endif

    printf("device                : %s\n", p.name);
    printf("compute capability    : %d.%d\n", p.major, p.minor);
    printf("multiProcessorCount   : %d\n", p.multiProcessorCount);
    printf("\n--- shared memory attributes (bytes / KiB) ---\n");
    printf("sharedMemPerBlock           (prop) : %zu  (%.2f KiB)\n", p.sharedMemPerBlock, p.sharedMemPerBlock/1024.0);
    printf("sharedMemPerBlockOptin      (prop) : %zu  (%.2f KiB)\n", p.sharedMemPerBlockOptin, p.sharedMemPerBlockOptin/1024.0);
    printf("sharedMemPerMultiprocessor  (prop) : %zu  (%.2f KiB)\n", p.sharedMemPerMultiprocessor, p.sharedMemPerMultiprocessor/1024.0);
    printf("\n--- via cudaDeviceGetAttribute ---\n");
    printf("MaxSharedMemoryPerBlock         : %d  (%.2f KiB)\n", perBlock, perBlock/1024.0);
    printf("MaxSharedMemoryPerBlockOptin    : %d  (%.2f KiB)  <-- the carve-out gate\n", optin, optin/1024.0);
    printf("MaxSharedMemoryPerMultiprocessor: %d  (%.2f KiB)\n", perSM, perSM/1024.0);
    printf("ReservedSharedMemoryPerBlock    : %d  (%.2f KiB)\n", reservedPerBlock, reservedPerBlock/1024.0);

    // Probe: how much dynamic smem can we actually opt a kernel into?
    printf("\n--- Tessera pin check ---\n");
    printf("gpu_target.py _SMEM_BYTES[SM_120] = 102400 (100 KiB)\n");
    printf("measured PerBlockOptin            = %d (%.2f KiB)\n", optin, optin/1024.0);
    printf("match(100KiB=102400)?             = %s\n", optin==102400 ? "YES" : "NO");
    return 0;
}
