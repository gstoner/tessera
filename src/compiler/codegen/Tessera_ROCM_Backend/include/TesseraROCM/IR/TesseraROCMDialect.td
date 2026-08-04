include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/OpBase.td"

def TesseraROCMDialect : Dialect {
  let name = "tessera_rocm";
  let cppNamespace = "mlir::tessera_rocm";
  let useDefaultTypePrinterParser = 1;
}

class ROCMStringAttr<string summaryText>
    : StringBasedAttr<StrAttrPred, summaryText>;

def MemSpaceAttr : ROCMStringAttr<"tessera_rocm.memspace">;
def TokenType : TypeDef<TesseraROCMDialect, "Token"> {
  let mnemonic = "token";
}
def ROCM_KernelAttr : ROCMStringAttr<"tessera_rocm.kernel">;

//===----------------------------------------------------------------------===//
// Semantic string attributes — Decision #21a (W1.1b)
//
// A `StrAttr` that SELECTS SEMANTICS must state its legal set, so an
// unrecognised value is rejected at parse/verify rather than falling through a
// chain of `==` comparisons to whatever the last `else` happens to do. Every
// ROCm kernel op previously took a bare `StrAttr` for `dtype`, `reduction` and
// `mode`, and the passes read them as free strings.
//
// WHAT THIS IS *NOT* APPLIED TO, and why the plan's count was wrong.
// W1.1b is written as "62 x `$name`", but `$name` is the emitted KERNEL SYMBOL
// -- measured values include "flash", "fc1", "fc2", "bwd", "acc",
// "default_int4_wmma". That is an open set chosen by the caller, not a
// semantic selector, and enumerating it would be incorrect. The genuinely
// semantic attributes are `dtype` (24 ops), `kind` (14), `reduction` (3),
// `mode`, `storage`, `route`, `activation`, `feature_map`, `counter`.
//
// This follows `EBM_EnumStrAttr` (adopted in W0.1 for exactly this purpose),
// which validates while keeping the textual form `dtype = "f32"`. A true
// `EnumAttr` would additionally give typed C++ accessors, but it changes the
// printed syntax and would break 34 lit fixtures, ~69 C++ sites and the Python
// Target-IR emitters at once -- so it is a separate, non-mechanical step.
//===----------------------------------------------------------------------===//

class ROCM_EnumStrAttr<list<string> cases, string summaryText>
    : StringBasedAttr<
        CPred<"::llvm::is_contained(::llvm::ArrayRef<::llvm::StringRef>({"
              # !interleave(!foreach(c, cases, "\"" # c # "\""), ", ") #
              "}), ::llvm::cast<::mlir::StringAttr>($_self).getValue())">,
        summaryText>;

// Storage dtype for a generated kernel.
//
// TWO constraints, not one union. A single shared attribute over every dtype
// any ROCm op accepts would admit `tessera_rocm.softmax` with `dtype = "int8"`
// -- a real dtype, wrong for that op -- and leave the rejection to the kernel
// generator, which is the very layering this item exists to remove. Measured
// across the 21 ops that carry `$dtype`: 20 accept only floats, and exactly one
// (`compare`) also accepts integers.
//
// The first cut of this file declared the union AND included fp8/int4/ue4m3/
// fp64/none. None of those is ever read from `$dtype`: they belong to
// `storage` / `index_storage` / `table_storage` on the quantisation ops, and
// they reached this list from a grep contaminated by those other attributes.
// `fpquant` is the instructive case -- its `$dtype` is the INPUT storage dtype
// (f32/f16/bf16); the target quant format rides on `max_normal` /
// `mantissa_bits` / `min_exp`.
//
// SEVEN ALIAS PAIRS remain admitted, which the bare `StrAttr` hid: f32/float32
// (canonical fp32), f16/float16 (canonical fp16), bf16/bfloat16, and on the
// numeric side i32/int32, u32/uint32. The ODS defaulted to one spelling and the
// passes accept several -- a live Decision #15a problem. They are named as
// aliases and the set is a shrink-only ratchet.

def ROCM_FloatDTypeAttr
    : ROCM_EnumStrAttr<["f32", "float32", "f16", "float16", "bf16", "bfloat16"],
                       "ROCm float kernel dtype: f32/f16/bf16 (or their "
                       "float32/float16/bfloat16 aliases)">;

// `compare` produces a boolean from either floats or integers, so it is the
// one op whose dtype set is genuinely wider. Stated separately rather than
// widening the shared one, so the extra values are legal for exactly the op
// that implements them.
def ROCM_NumericDTypeAttr
    : ROCM_EnumStrAttr<["f32", "float32", "f16", "float16", "bf16", "bfloat16",
                        "i32", "int32", "u32", "uint32"],
                       "ROCm numeric kernel dtype: the float set plus "
                       "i32/int32 and u32/uint32">;

// The WMMA GEMM is the one op whose dtype set is about the MATRIX-CORE
// instruction rather than about floats: gfx1151 WMMA has f16/bf16 and iu8/iu4
// variants, so int8 and int4 are first-class GEMM storage dtypes here (and
// exactly the reason `int8` must NOT be legal on `softmax`, whose generator has
// no integer path). Mirrors `target_ir.py`'s gate value-for-value, including
// its `f16` default -- fp32 is deliberately absent, matching both that gate and
// the op's own ODS default.
//
// This constraint is why the two-way split was wrong: the first attempt at
// fixing the review comment derived the float set from `Generate*Kernel.cpp`
// `dt == "..."` chains, which the GEMM does not use, so it silently narrowed
// the int8/int4 lane out of existence -- the compiled int8 matmul stopped
// verifying and fell back to the f16/bf16 executor. Caught by the execution
// tests, not by the ODS ones, which is the argument for keeping both.
def ROCM_GemmDTypeAttr
    : ROCM_EnumStrAttr<["f16", "float16", "bf16", "bfloat16",
                        "int8", "i8", "int4", "i4"],
                       "ROCm WMMA GEMM dtype: f16/bf16 or the iu8/iu4 "
                       "integer matrix-core widths">;

def ROCM_ReductionAttr
    : ROCM_EnumStrAttr<["none", "sum", "mean"],
                       "ROCm reduction policy: 'none', 'sum', or 'mean'">;

// Scatter/segment combine mode.
// Scatter/segment combine mode.
//
// `set` was missing from the first cut of this list, which made the compiler
// REJECT a valid program -- the dangerous direction for a constraint. It came
// from deriving the legal set by grepping `mode == "..."` comparisons instead
// of reading the pass that consumes it, and the producers build these values
// by string interpolation so no static scan could have found it either.
def ROCM_CombineModeAttr
    : ROCM_EnumStrAttr<["add", "max", "min", "set", "weighted_add"],
                       "ROCm combine mode: 'add', 'max', 'min', 'set', "
                       "'weighted_add'">;
