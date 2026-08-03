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
// THREE spellings were in play, which is a Decision #15a problem the bare
// `StrAttr` hid: the ODS defaulted to "f32"/"f16", the passes accepted "f16"
// OR "float16" and "bf16" OR "bfloat16", and `tessera.dtype`'s canonical names
// are "fp32"/"fp16"/"bf16". All are admitted here so nothing breaks, and the
// aliases are named as aliases -- normalising the tree onto the canonical
// spelling is follow-on work, gated by `test_rocm_dtype_attr_contract.py`.
def ROCM_DTypeAttr
    : ROCM_EnumStrAttr<[// canonical (tessera.dtype)
                        "fp32", "fp16", "bf16", "fp64",
                        "fp8_e4m3", "fp8_e5m2", "int8", "int4", "int32",
                        // ALIASES the ROCm passes accept today. Seven pairs:
                        // f32/float32/fp32, f16/float16/fp16, bf16/bfloat16,
                        // i8/int8, i4/int4, i32/int32, u32/uint32. Admitting
                        // them keeps the tree building; the set is a
                        // shrink-only ratchet.
                        "f32", "float32", "f16", "float16", "bfloat16",
                        "i8", "i4", "i32", "u32", "uint32",
                        // microscaling SCALE dtypes -- not storage dtypes, but
                        // carried on the same attribute by the quant kernels
                        "ue4m3", "ue8m0",
                        // explicit "no dtype" for ops that key off operands
                        "none"],
                       "ROCm kernel storage dtype">;

// Reduction policy. Selects semantics (what the kernel computes), so it fails
// closed rather than defaulting.
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
