# Build for consumer Blackwell sm_120 (arch-specific: FP4 block-scale needs 120a).
# On SuperBear (WSL2, CUDA 13.3 toolkit-only). If nvcc rejects the s4 MMA for
# sm_120a, run `make NO_INT4=1` — probe/bench then report the INT4 path as
# compiled_out instead of failing the build (plan §Phase 0).
NVCC   ?= nvcc
ARCH   ?= sm_120a
FLAGS   = -O3 -std=c++17 -arch=$(ARCH) --expt-relaxed-constexpr
ifdef NO_INT4
FLAGS  += -DNO_INT4_MMA
endif

all: probe bench stage_probe

probe: probe.cu
	$(NVCC) $(FLAGS) -o $@ $<

bench: bench.cu
	$(NVCC) $(FLAGS) -o $@ $< -lcublasLt

stage_probe: stage_probe.cu
	$(NVCC) $(FLAGS) -o $@ $<

clean:
	rm -f probe bench stage_probe

# convenience: prove the toolchain accepts the arch before anything else
archcheck:
	@echo 'int main(){return 0;}' > /tmp/_a.cu && $(NVCC) -arch=$(ARCH) /tmp/_a.cu -o /tmp/_a \
	  && echo "OK: $(ARCH) accepted" || echo "FAIL: toolchain rejects $(ARCH)"
