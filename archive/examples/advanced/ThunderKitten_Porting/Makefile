
.PHONY: build bench report tune clean
BUILD_DIR ?= build
OUT_DIR ?= out
COPY_PATH ?= cp.async     # cp.async | tma
COMPUTE_PATH ?= wmma      # wmma | wgmma
SHAPE ?= M=4096,N=4096,K=4096

build:
	@mkdir -p $(BUILD_DIR)
	@cmake -S . -B $(BUILD_DIR) -DTESSERA_USE_WGMMA=$$( [ "$(COMPUTE_PATH)" = "wgmma" ] && echo ON || echo OFF ) -DTESSERA_USE_TMA=$$( [ "$(COPY_PATH)" = "tma" ] && echo ON || echo OFF )
	@cmake --build $(BUILD_DIR) -j

bench: build
	@mkdir -p $(OUT_DIR)
	@$(BUILD_DIR)/tessera/tools/porting/microbench/tessera_microbench \
		--M 4096 --N 4096 --K 4096 \
		--block_m 128 --block_n 128 --block_k 32 \
		--stages 2 --split_k 1 --cta_pairs 0 \
		--copy_path $(COPY_PATH) --compute_path $(COMPUTE_PATH) \
		--json_out $(OUT_DIR)/bench.json
	@python3 - <<'PY'\
import json, csv; d=json.load(open('out/bench.json')); \
w=csv.writer(open('out/results.csv','w',newline='')); \
w.writerow(['arch','op','M','N','K','BM','BN','BK','stages','split_k','cta_pairs','copy_path','compute_path','ms_avg','tflops']); \
w.writerow([d['arch'],'tile_gemm',d['M'],d['N'],d['K'],d['BM'],d['BN'],d['BK'],d['stages'],d['split_k'],d['cta_pairs'],d['copy_path'],d['compute_path'],d['ms_avg'],d['tflops']])
PY

report:
	@python3 tessera/tools/porting/scripts/generate_report.py --csv $(OUT_DIR)/results.csv --out $(OUT_DIR)/report
	@echo "Report at $(OUT_DIR)/report/index.html"

tune: build
	@mkdir -p $(OUT_DIR)
	@python3 tessera/tools/porting/autotune/hyperband.py \
		--db $(OUT_DIR)/sched.db --out $(OUT_DIR)/best.json \
		--arch sm90 --dtype bf16 --op gemm --shape $(SHAPE) \
		--exe $(BUILD_DIR)/tessera/tools/porting/microbench/tessera_microbench \
		--copy_path $(COPY_PATH) --compute_path $(COMPUTE_PATH)

clean:
	@rm -rf $(BUILD_DIR) $(OUT_DIR)


bench_rope: build
	@mkdir -p $(OUT_DIR)
	@$(BUILD_DIR)/tessera/tools/porting/rope/tessera_rope_bench --S 4096 --D 128 --theta 10000 > $(OUT_DIR)/rope.json
	@echo "RoPE results -> $(OUT_DIR)/rope.json"


# v11.1: Example WGMMA correctness check target (placeholder harness)
check_wgmma_map: build
	@$(BUILD_DIR)/tessera/tools/porting/tests/tessera_wgmma_map_check

# Epilogue example
bench_epilogue: build
	@mkdir -p $(OUT_DIR)
	@$(BUILD_DIR)/tessera/tools/porting/microbench/tessera_microbench \
		--M 4096 --N 4096 --K 4096 \
		--block_m 128 --block_n 128 --block_k 32 \
		--stages 2 --split_k 1 --cta_pairs 0 \
		--copy_path $(COPY_PATH) --compute_path $(COMPUTE_PATH) \
		--epilogue bias_gelu --json_out $(OUT_DIR)/bench_epilogue.json
