#include "tessera_runtime.h"
#include <stdio.h>
#include <string.h>

#define CHK(x) do{ tessStatus_t s=(x); if(s!=TESS_OK){ \
  fprintf(stderr,"ERROR %s:%d: %s\n", __FILE__, __LINE__, tessStatusString(s)); return 1; } }while(0)

typedef struct {
  void* Q; void* K; void* V; void* O;
  int B,H,L,D;
  float scale;
} FlashArgs;

int main() {
  tessContext_t ctx; CHK(tessInit(&ctx));
  int devs[8]={0,1,2,3,4,5,6,7};
  tessMesh_t mesh; CHK(tessMeshCreate(ctx, devs, 8, (tessMeshAxes){2,1,4,0}, &mesh));
  tessStream_t s; CHK(tessStreamCreate(mesh, 0, &s));

  void *Q,*K,*V,*O;
  CHK(tessMalloc(mesh, 1024, &Q));
  CHK(tessMalloc(mesh, 1024, &K));
  CHK(tessMalloc(mesh, 1024, &V));
  CHK(tessMalloc(mesh, 1024, &O));

  const unsigned char fatbin[4] = {0xCA,0xFE,0xBA,0xBE};
  tessModule_t mod; CHK(tessModuleLoad(ctx, fatbin, sizeof(fatbin), &mod));
  tessKernel_t ker; CHK(tessKernelGet(mod, "flash_attention_fused", &ker));

  FlashArgs args = {Q,K,V,O, 8,16,4096,256, 1.0f};
  tessLaunchConfig cfg = { {256,1,1}, {128,1,1}, 0, TESS_LAUNCH_DETERMINISTIC };
  CHK(tessLaunch(ker, mesh, cfg, &args, sizeof(args), s));

  CHK(tessStreamSynchronize(s));
  CHK(tessStreamDestroy(s));
  CHK(tessModuleUnload(mod));
  CHK(tessFree(mesh, Q)); CHK(tessFree(mesh, K)); CHK(tessFree(mesh, V)); CHK(tessFree(mesh, O));
  CHK(tessMeshDestroy(mesh));
  CHK(tessShutdown(ctx));
  printf("tessera_demo: OK\\n");
  return 0;
}
