// Spike #6 — load Tessera-emit-style PTX via the CUDA Driver API and launch it on sm_120.
// This is the faithful "emit PTX -> ptxas (JIT) -> load -> launch -> compare" path.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda.h>

#define DK(x) do{CUresult r=(x); if(r){const char*s; cuGetErrorString(r,&s); printf("CU err %s @%d: %s\n",#x,__LINE__,s); exit(1);} }while(0)

#define M 16
#define N 8
#define K 16

// Minimal bf16 (round-to-nearest-even) packed in uint16.
static unsigned short f2bf16(float f){
    unsigned int x; __builtin_memcpy(&x,&f,4);
    unsigned int r = (x>>16)&1; unsigned int t = 0x7fff + r;
    x += t; return (unsigned short)(x>>16);
}
static float bf162f(unsigned short h){ unsigned int x=((unsigned int)h)<<16; float f; __builtin_memcpy(&f,&x,4); return f; }

int main(int argc, char** argv){
    const char* ptxfile = argc>1?argv[1]:"tessera_mma_m16n8k16.ptx";
    FILE* fp=fopen(ptxfile,"rb"); if(!fp){printf("cannot open %s\n",ptxfile);return 1;}
    fseek(fp,0,SEEK_END); long sz=ftell(fp); fseek(fp,0,SEEK_SET);
    std::vector<char> ptx(sz+1); fread(ptx.data(),1,sz,fp); ptx[sz]=0; fclose(fp);

    DK(cuInit(0));
    CUdevice dev; DK(cuDeviceGet(&dev,0));
    char name[128]; DK(cuDeviceGetName(name,128,dev));
    int maj,min; DK(cuDeviceGetAttribute(&maj,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,dev));
    DK(cuDeviceGetAttribute(&min,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,dev));
    CUcontext ctx; DK(cuDevicePrimaryCtxRetain(&ctx,dev)); DK(cuCtxSetCurrent(ctx));

    // JIT-compile the PTX (ptxas runs inside the driver) for the live device.
    char log[8192]; log[0]=0;
    CUjit_option opts[]={CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES};
    void* ovals[]={(void*)log, (void*)(size_t)sizeof(log)};
    CUmodule mod;
    CUresult r = cuModuleLoadDataEx(&mod, ptx.data(), 2, opts, ovals);
    if(r){ printf("cuModuleLoadDataEx failed: %s\n", log); return 1; }
    CUfunction fn; DK(cuModuleGetFunction(&fn,mod,"tessera_mma_m16n8k16_bf16"));

    // Host data: A row-major bf16, B col-major bf16.
    std::vector<unsigned short> hA(M*K), hB(K*N);
    std::vector<float> fA(M*K), fB(K*N), hD(M*N,0), ref(M*N,0);
    srand(1234);
    for(int i=0;i<M*K;i++){ float v=((rand()%201)-100)/100.0f; hA[i]=f2bf16(v); fA[i]=bf162f(hA[i]); }
    for(int i=0;i<K*N;i++){ float v=((rand()%201)-100)/100.0f; hB[i]=f2bf16(v); fB[i]=bf162f(hB[i]); }
    for(int m=0;m<M;m++) for(int n=0;n<N;n++){ float a=0; for(int k=0;k<K;k++) a+=fA[m*K+k]*fB[k+n*K]; ref[m*N+n]=a; }

    CUdeviceptr dA,dB,dD;
    DK(cuMemAlloc(&dA,M*K*2)); DK(cuMemAlloc(&dB,K*N*2)); DK(cuMemAlloc(&dD,M*N*4));
    DK(cuMemcpyHtoD(dA,hA.data(),M*K*2));
    DK(cuMemcpyHtoD(dB,hB.data(),K*N*2));
    void* args[]={&dA,&dB,&dD};
    DK(cuLaunchKernel(fn, 1,1,1, 32,1,1, 0, 0, args, 0));
    DK(cuCtxSynchronize());
    DK(cuMemcpyDtoH(hD.data(),dD,M*N*4));

    double maxabs=0; int bad=0;
    for(int i=0;i<M*N;i++){ double e=fabs(hD[i]-ref[i]); if(e>maxabs)maxabs=e; if(e>1e-2)bad++; }
    printf("device: %s (CC %d.%d), loaded %s via Driver API (cuModuleLoadDataEx)\n", name,maj,min,ptxfile);
    printf("max abs error vs CPU ref = %g, elements over 1e-2 = %d / %d\n", maxabs,bad,M*N);
    printf("sample D[0..3] = %.4f %.4f %.4f %.4f   ref = %.4f %.4f %.4f %.4f\n",
           hD[0],hD[1],hD[2],hD[3], ref[0],ref[1],ref[2],ref[3]);
    printf("RESULT: %s\n", bad==0?"PASS (emitted PTX assembled+launched+matched reference)":"FAIL");
    return bad==0?0:1;
}
