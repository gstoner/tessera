// Strided-copy / gather kernel (f32) for the Tessera x86 backend — the 0-move
// lane (P4 of S_SERIES_GAP_CLOSURE_PLAN): pad / cat / roll / flip / tile /
// repeat / stack all reduce to `out[i] = src[idx[i]]` for a per-op index map.
//
// The host computes the integer index map by running the op's numpy semantics
// on an arange index grid (shape arithmetic only — it never touches the f32
// data); this kernel does the actual data movement on-device. A negative index
// is a sentinel meaning "leave out[i] unchanged", so the caller pre-initializes
// `out` (to the pad fill value, or 0 for a cat-accumulate over per-input passes)
// and a single masked gather realizes pad's borders and cat's per-input writes.

#include <cstdint>

extern "C" void tessera_x86_gather_f32(const float* src, int64_t src_n,
                                       const int64_t* idx, int64_t n,
                                       float* out) {
    for (int64_t i = 0; i < n; ++i) {
        int64_t j = idx[i];
        if (j >= 0 && j < src_n)
            out[i] = src[j];   // else: leave out[i] (pad fill / cat keep)
    }
}
