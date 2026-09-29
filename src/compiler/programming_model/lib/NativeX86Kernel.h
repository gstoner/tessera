// E2E-REAL-6, x86 elementwise / cohort-2 / breadth families (2026-09-28).
//
// One isolated static x86 Graph operation -> a content-addressed Schedule
// record -> the Tile launch envelope the stable x86 C ABI consumes. This is
// the native owner of what `x86_native.{package_elementwise, package_cohort2}`
// and `x86_breadth.package_graph_breadth` used to decide in Python by
// authoring Tile IR text beside the compiled route. The retired Python
// constructors survive only as the declared differential oracle
// (`tests/_support/x86_kernel_baseline.py`, Decision #31(a)).
//
// Ownership is opt-in per module: the Graph must name `tessera.target = "x86"`,
// `tessera.arch = "zen5-avx512"` and carry `tessera.launch_bindings` (the host
// buffer aliases, inputs then output). Without the bindings attribute these
// ops pass through Graph -> Schedule untouched, exactly as before; with it,
// anything outside the envelope below fails closed (Decision #21a): unknown
// attributes, non-static shapes, the wrong element types, a result that is not
// the function's only return.
//
// `tessera.absolute`/`floor`/`ceil`/`cumsum` stay with NativeAbsolute.h (their
// contract predates this one); normalization rides the semantic-kernel
// `schedule.norm` route; `tessera.alibi` stays on its retired route because
// its Graph operand list is not decodable by position (ODS declares no slopes
// operand; `tests/unit/test_op_arity_contract.py::_UNDECODABLE_OPERAND_LISTS`).
namespace {

struct X86KernelSpec {
  StringRef op;
  StringRef family;     // elementwise | argreduce | scan | rope | abi
  StringRef subfamily;  // elementwise ABI family, or the breadth ABI key
  StringRef kind;
};

// Every op this contract owns. The breadth ABI keys, symbols and ABI ids must
// agree with `python/tessera/compiler/x86_breadth.py::X86_BREADTH_ABIS`
// (pinned by `tests/unit/test_x86_kernel_differential.py`).
static const X86KernelSpec kX86KernelSpecs[] = {
    {"tessera.sqrt", "elementwise", "unary", "sqrt"},
    {"tessera.rsqrt", "elementwise", "unary", "rsqrt"},
    {"tessera.reciprocal", "elementwise", "unary", "reciprocal"},
    {"tessera.sign", "elementwise", "unary", "sign"},
    {"tessera.round", "elementwise", "unary", "round"},
    {"tessera.sub", "elementwise", "binary", "sub"},
    {"tessera.div", "elementwise", "binary", "div"},
    {"tessera.maximum", "elementwise", "binary", "maximum"},
    {"tessera.minimum", "elementwise", "binary", "minimum"},
    {"tessera.add", "elementwise", "binary", "add"},
    {"tessera.mul", "elementwise", "binary", "mul"},
    {"tessera.mod", "elementwise", "binary", "mod"},
    {"tessera.floor_div", "elementwise", "binary", "floor_div"},
    {"tessera.isnan", "elementwise", "predicate", "isnan"},
    {"tessera.isinf", "elementwise", "predicate", "isinf"},
    {"tessera.isfinite", "elementwise", "predicate", "isfinite"},
    {"tessera.eq", "elementwise", "compare", "eq"},
    {"tessera.ne", "elementwise", "compare", "ne"},
    {"tessera.lt", "elementwise", "compare", "lt"},
    {"tessera.le", "elementwise", "compare", "le"},
    {"tessera.gt", "elementwise", "compare", "gt"},
    {"tessera.ge", "elementwise", "compare", "ge"},
    {"tessera.logical_and", "elementwise", "logical", "and"},
    {"tessera.logical_or", "elementwise", "logical", "or"},
    {"tessera.logical_xor", "elementwise", "logical", "xor"},
    {"tessera.logical_not", "elementwise", "logical", "not"},
    {"tessera.bitwise_and", "elementwise", "bitwise", "and"},
    {"tessera.bitwise_or", "elementwise", "bitwise", "or"},
    {"tessera.bitwise_xor", "elementwise", "bitwise", "xor"},
    {"tessera.bitwise_not", "elementwise", "bitwise", "not"},
    {"tessera.popcount", "elementwise", "bitwise", "popcount"},
    {"tessera.where", "elementwise", "where", "where"},
    {"tessera.exp", "elementwise", "transcendental", "exp"},
    {"tessera.log", "elementwise", "transcendental", "log"},
    {"tessera.tanh", "elementwise", "transcendental", "tanh"},
    {"tessera.sigmoid", "elementwise", "transcendental", "sigmoid"},
    {"tessera.silu", "elementwise", "transcendental", "silu"},
    {"tessera.gelu", "elementwise", "transcendental", "gelu"},
    {"tessera.erf", "elementwise", "transcendental", "erf"},
    {"tessera.softplus", "elementwise", "transcendental", "softplus"},
    {"tessera.expm1", "elementwise", "transcendental", "expm1"},
    {"tessera.log1p", "elementwise", "transcendental", "log1p"},
    {"tessera.cos", "elementwise", "transcendental", "cos"},
    {"tessera.tan", "elementwise", "transcendental", "tan"},
    {"tessera.sinh", "elementwise", "transcendental", "sinh"},
    {"tessera.cosh", "elementwise", "transcendental", "cosh"},
    {"tessera.asin", "elementwise", "transcendental", "asin"},
    {"tessera.acos", "elementwise", "transcendental", "acos"},
    {"tessera.atan", "elementwise", "transcendental", "atan"},
    {"tessera.erfc", "elementwise", "transcendental", "erfc"},
    {"tessera.sin", "elementwise", "transcendental", "sin"},
    {"tessera.lgamma", "elementwise", "transcendental", "lgamma"},
    {"tessera.digamma", "elementwise", "transcendental", "digamma"},
    {"tessera.pow", "elementwise", "binary_math", "pow"},
    {"tessera.silu_mul", "elementwise", "binary_math", "silu_mul"},
    {"tessera.argmax", "argreduce", "argreduce", "argmax"},
    {"tessera.argmin", "argreduce", "argreduce", "argmin"},
    {"tessera.cumprod", "scan", "scan", "product"},
    {"tessera.cummax", "scan", "scan", "max"},
    {"tessera.cummin", "scan", "scan", "min"},
    {"tessera.rope", "rope", "rope", "rope"},
    {"tessera.gather", "abi", "gather_f32", "gather"},
    {"tessera.loss.mse", "abi", "pointwise_loss_f32", "pointwise_loss"},
    {"tessera.loss.mae", "abi", "pointwise_loss_f32", "pointwise_loss"},
    {"tessera.loss.huber", "abi", "pointwise_loss_f32", "pointwise_loss"},
    {"tessera.loss.smooth_l1", "abi", "pointwise_loss_f32", "pointwise_loss"},
    {"tessera.loss.log_cosh", "abi", "pointwise_loss_f32", "pointwise_loss"},
    {"tessera.cholesky", "abi", "cholesky_f32", "cholesky"},
    {"tessera.tri_solve", "abi", "tri_solve_f32", "tri_solve"},
};

static const X86KernelSpec *x86KernelSpec(Operation *op) {
  StringRef name = op->getName().getStringRef();
  for (const X86KernelSpec &spec : kX86KernelSpecs)
    if (spec.op == name)
      return &spec;
  return nullptr;
}

// One breadth C ABI entry: symbol, ABI id, x86 pipeline family, effects, and
// the ordered argument list ("b:<name>:<storage>:<direction>" buffers,
// "s:<name>:<mlir type>" scalars).
struct X86AbiSpec {
  StringRef key;
  StringRef symbol;
  StringRef abi;
  StringRef family;
  StringRef effects;
  ArrayRef<StringRef> args;
};

static const StringRef kGatherArgs[] = {"b:source:f32:input", "s:SourceN:i64",
                                        "b:indices:i64:input", "s:N:i64",
                                        "b:output:f32:output"};
static const StringRef kLossArgs[] = {"b:prediction:f32:input",
                                      "b:target:f32:input", "s:N:i64",
                                      "s:Kind:i32", "s:Parameter:f32",
                                      "b:output:f32:output"};
static const StringRef kCholeskyArgs[] = {"b:matrix:f32:input", "s:Batch:i64",
                                          "s:N:i64", "b:lower:f32:output"};
static const StringRef kTriSolveArgs[] = {"b:matrix:f32:input", "b:rhs:f32:input",
                                          "s:Batch:i64", "s:N:i64", "s:M:i64",
                                          "s:Lower:i32", "b:output:f32:output"};

static const X86AbiSpec kX86AbiSpecs[] = {
    {"gather_f32", "tessera_x86_gather_f32", "tessera.x86.gather.f32.v1",
     "movement", "writeonly", kGatherArgs},
    {"pointwise_loss_f32", "tessera_x86_avx512_pointwise_loss_f32",
     "tessera.x86.pointwise.loss.f32.v1", "loss", "writeonly", kLossArgs},
    {"cholesky_f32", "tessera_x86_cholesky_f32", "tessera.x86.cholesky.f32.v1",
     "linalg", "writeonly", kCholeskyArgs},
    {"tri_solve_f32", "tessera_x86_tri_solve_f32",
     "tessera.x86.tri.solve.f32.v1", "linalg", "writeonly", kTriSolveArgs},
};

static const X86AbiSpec *x86AbiSpec(StringRef key) {
  for (const X86AbiSpec &spec : kX86AbiSpecs)
    if (spec.key == key)
      return &spec;
  return nullptr;
}

// Storage of one binding. `i1` tensors are byte-per-element bool buffers.
static StringRef x86Storage(Type element) {
  if (element.isF32()) return "f32";
  if (element.isInteger(1)) return "i8";
  if (element.isInteger(32)) return "i32";
  if (element.isInteger(64)) return "i64";
  return {};
}

// The element storage one elementwise ABI family expects for its inputs and
// its output.
static std::pair<StringRef, StringRef> elementwiseStorage(StringRef sub) {
  if (sub == "logical") return {"i8", "i8"};
  if (sub == "bitwise") return {"i32", "i32"};
  if (sub == "predicate" || sub == "compare") return {"f32", "i8"};
  return {"f32", "f32"};
}

static unsigned elementwiseArity(StringRef sub, StringRef kind) {
  if (sub == "where") return 3;
  if (sub == "binary" || sub == "compare" || sub == "binary_math") return 2;
  if (sub == "logical") return kind == "not" ? 1 : 2;
  if (sub == "bitwise") return kind == "not" || kind == "popcount" ? 1 : 2;
  return 1;
}

// The pointwise-loss ABI's Kind selector (x86_breadth._POINTWISE_LOSSES).
static int lossKind(StringRef op) {
  return llvm::StringSwitch<int>(op)
      .Case("tessera.loss.mse", 0)
      .Case("tessera.loss.mae", 1)
      .Case("tessera.loss.huber", 2)
      .Case("tessera.loss.smooth_l1", 3)
      .Default(4);  // tessera.loss.log_cosh
}

static std::string x86KernelHash(DictionaryAttr contract) {
  std::string text;
  llvm::raw_string_ostream os(text);
  contract.print(os);
  os.flush();
  return llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(text)), true);
}

// Derive the contract for one owned Graph op, or fail with a diagnostic. The
// same function re-derives it at Schedule -> Tile, so a Schedule record that
// no longer matches its Graph parent is refused.
static FailureOr<DictionaryAttr> x86KernelContract(Operation *op) {
  const X86KernelSpec *spec = x86KernelSpec(op);
  auto fn = op->getParentOfType<func::FuncOp>();
  auto mod = op->getParentOfType<ModuleOp>();
  if (!spec || !fn || !mod)
    return op->emitError("x86 native kernel requires a registered owned op"),
           failure();
  auto target = mod->getAttrOfType<StringAttr>("tessera.target");
  auto arch = mod->getAttrOfType<StringAttr>("tessera.arch");
  if (!target || target.getValue() != "x86" || !arch ||
      arch.getValue() != "zen5-avx512")
    return op->emitError("x86 native kernel requires the zen5-avx512 x86 target"),
           failure();
  if (!fn.getBody().hasOneBlock() || op->getNumResults() != 1 ||
      fn.getNumResults() != 1 || op->getNumOperands() == 0 ||
      fn.getNumArguments() == 0 || fn.getNumArguments() > op->getNumOperands())
    return op->emitError("x86 native kernel requires an isolated one-result "
                         "entry whose arguments are its operands"),
           failure();
  // Every operand is an entry argument (an argument may repeat: add(x, x)),
  // and every entry argument is used.
  llvm::SmallDenseSet<unsigned> used;
  for (Value operand : op->getOperands()) {
    auto arg = dyn_cast<BlockArgument>(operand);
    if (!arg || arg.getOwner() != &fn.getBody().front())
      return op->emitError("x86 native kernel operands must be entry arguments"),
             failure();
    used.insert(arg.getArgNumber());
  }
  if (used.size() != fn.getNumArguments())
    return op->emitError("x86 native kernel entry has an unused argument"),
           failure();
  // Argument attributes: the frontend's dimension names and row-major layout
  // are the only argument policy this ABI implements.
  for (unsigned index = 0; index < fn.getNumArguments(); ++index) {
    auto ty = dyn_cast<RankedTensorType>(fn.getArgument(index).getType());
    if (auto attrs = fn.getArgAttrDict(index))
      for (NamedAttribute attr : attrs) {
        if (attr.getName() == "tessera.layout") {
          auto layout = dyn_cast<StringAttr>(attr.getValue());
          if (!layout || layout.getValue() != "row_major")
            return op->emitError("x86 native kernel requires row_major arguments"),
                   failure();
          continue;
        }
        auto names = dyn_cast<ArrayAttr>(attr.getValue());
        if (attr.getName() != "tessera.dim_names" || !names || !ty ||
            names.size() != static_cast<size_t>(ty.getRank()))
          return op->emitError("x86 native kernel argument policy is unsupported"),
                 failure();
      }
  }

  SmallVector<RankedTensorType> types;
  for (Value value : op->getOperands()) {
    auto ty = dyn_cast<RankedTensorType>(value.getType());
    if (!ty || !ty.hasStaticShape() || ty.getRank() < 1 || ty.getEncoding() ||
        llvm::any_of(ty.getShape(), [](int64_t d) { return d <= 0; }))
      return op->emitError("x86 native kernel requires static positive-extent "
                           "operands"),
             failure();
    types.push_back(ty);
  }
  auto result = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!result || !result.hasStaticShape() || result.getEncoding() ||
      fn.getResultTypes()[0] != result ||
      llvm::any_of(result.getShape(), [](int64_t d) { return d <= 0; }))
    return op->emitError("x86 native kernel requires a static positive result"),
           failure();
  types.push_back(result);

  auto names = mod->getAttrOfType<ArrayAttr>("tessera.launch_bindings");
  if (!names || names.size() != types.size() ||
      llvm::any_of(names, [](Attribute a) {
        auto s = dyn_cast<StringAttr>(a);
        return !s || s.getValue().empty();
      }))
    return op->emitError("x86 native kernel requires one binding name per "
                         "operand and result"),
           failure();
  for (unsigned index = 0; index + 1 < names.size(); ++index)
    if (names[index] == names[names.size() - 1])
      return op->emitError("x86 native kernel output binding must be distinct"),
             failure();

  OpBuilder b(op);
  SmallVector<NamedAttribute> semantic;  // family-specific contract keys
  SmallVector<NamedAttribute> scalars;   // the ABI's compile-time scalars
  llvm::StringSet<> allowed{"tessera.effect_kind", "schedule.artifact_hash"};
  auto i64 = [&](int64_t v) { return b.getI64IntegerAttr(v); };
  auto elements = [](RankedTensorType ty) {
    int64_t n = 1;
    for (int64_t d : ty.getShape()) {
      if (n > std::numeric_limits<int64_t>::max() / d) return int64_t(-1);
      n *= d;
    }
    return n;
  };
  auto fail = [&](const Twine &message) -> FailureOr<DictionaryAttr> {
    op->emitError(message);
    return failure();
  };
  auto rowsOf = [&](RankedTensorType ty) {
    int64_t rows = 1;
    for (int64_t d : ty.getShape().drop_back()) rows *= d;
    return rows;
  };

  StringRef family = spec->family;
  if (family == "elementwise") {
    unsigned arity = elementwiseArity(spec->subfamily, spec->kind);
    auto [inStorage, outStorage] = elementwiseStorage(spec->subfamily);
    if (op->getNumOperands() != arity)
      return fail("x86 native elementwise has the wrong operand count");
    for (unsigned index = 0; index < arity; ++index) {
      StringRef want = spec->subfamily == "where" && index == 0 ? "i8" : inStorage;
      if (x86Storage(types[index].getElementType()) != want ||
          types[index].getShape() != result.getShape())
        return fail("x86 native elementwise requires same-shape operands of its "
                    "ABI storage");
    }
    if (x86Storage(result.getElementType()) != outStorage)
      return fail("x86 native elementwise result storage is unsupported");
    int64_t n = elements(result);
    if (n <= 0) return fail("x86 native elementwise shape exceeds its ABI");
    semantic.push_back(b.getNamedAttr("elementwise_family",
                                      b.getStringAttr(spec->subfamily)));
    scalars.push_back(b.getNamedAttr("N", i64(n)));
  } else if (family == "argreduce" || family == "scan") {
    RankedTensorType input = types[0];
    if (op->getNumOperands() != 1 || !input.getElementType().isF32())
      return fail("x86 native argreduce/scan requires one f32 operand");
    allowed.insert("axis");
    auto axisAttr = op->getAttrOfType<IntegerAttr>("axis");
    if (op->hasAttr("axis") && !axisAttr)
      return fail("x86 native argreduce/scan axis must be an integer");
    // argmax/argmin: an absent axis flattens the operand (NumPy `axis=None`,
    // the ODS contract). Scans default their axis to -1 in ODS, so an absent
    // scan axis is the last axis; a flattened scan is not expressible here.
    bool flatten = !axisAttr && family == "argreduce";
    int64_t n = elements(input);
    if (n <= 0) return fail("x86 native argreduce/scan shape exceeds its ABI");
    int64_t rows = flatten ? 1 : rowsOf(input);
    int64_t cols = flatten ? n : input.getShape().back();
    if (axisAttr) {
      int64_t axis = axisAttr.getInt();
      if (axis != -1 && axis != input.getRank() - 1)
        return fail("x86 native argreduce/scan requires the last axis");
    }
    SmallVector<int64_t> logical = flatten ? SmallVector<int64_t>{n}
                                           : SmallVector<int64_t>(input.getShape());
    if (family == "scan") {
      if (!result.getElementType().isF32() ||
          result.getShape() != ArrayRef<int64_t>(logical))
        return fail("x86 native scan result must be the f32 (flattened) input shape");
      semantic.push_back(b.getNamedAttr("inclusive", b.getBoolAttr(true)));
    } else {
      allowed.insert("keepdims");
      auto keepAttr = op->getAttrOfType<BoolAttr>("keepdims");
      if (op->hasAttr("keepdims") && !keepAttr)
        return fail("x86 native argreduce keepdims must be a boolean");
      bool keepdims = keepAttr && keepAttr.getValue();
      SmallVector<int64_t> expected(logical.begin(), logical.end() - 1);
      if (keepdims) expected.push_back(1);
      if (!result.getElementType().isInteger(32) ||
          result.getShape() != ArrayRef<int64_t>(expected))
        return fail("x86 native argreduce result must be i32 with its axis removed");
      semantic.push_back(b.getNamedAttr("keepdims", b.getBoolAttr(keepdims)));
      semantic.push_back(b.getNamedAttr("tie_break", b.getStringAttr("first")));
    }
    semantic.push_back(b.getNamedAttr("logical_shape", b.getDenseI64ArrayAttr(logical)));
    semantic.push_back(b.getNamedAttr(
        "axis_mode", b.getStringAttr(flatten ? "flatten" : "last")));
    scalars.push_back(b.getNamedAttr("Rows", i64(rows)));
    scalars.push_back(b.getNamedAttr("Cols", i64(cols)));
  } else if (family == "rope") {
    if (op->getNumOperands() != 2 || types[0] != types[1] || types[0] != result ||
        !result.getElementType().isF32() || result.getShape().back() % 2)
      return fail("x86 native rope requires same-shape f32 x/theta/output with "
                  "an even last dimension");
    semantic.push_back(b.getNamedAttr("pair_layout",
                                      b.getStringAttr("interleaved_pairs")));
    scalars.push_back(b.getNamedAttr("Rows", i64(rowsOf(result))));
    scalars.push_back(b.getNamedAttr("Cols", i64(result.getShape().back())));
  } else {
    // Breadth: an isomorphic public Graph op over one stable C ABI entry.
    const X86AbiSpec *abi = x86AbiSpec(spec->subfamily);
    if (!abi) return fail("x86 native breadth ABI is not registered");
    if (!result.getElementType().isF32())
      return fail("x86 native breadth requires an f32 result");
    StringRef name = op->getName().getStringRef();
    if (spec->kind == "gather") {
      allowed.insert("axis");
      auto axis = op->getAttrOfType<IntegerAttr>("axis");
      if (op->getNumOperands() != 2 || types[0].getRank() != 1 ||
          types[1].getRank() != 1 || !types[0].getElementType().isF32() ||
          !types[1].getElementType().isInteger(64) ||
          result.getShape() != types[1].getShape() ||
          (op->hasAttr("axis") && (!axis || (axis.getInt() != 0 && axis.getInt() != -1))))
        return fail("x86 native gather requires a rank-1 f32 source, rank-1 i64 "
                    "indices and axis 0");
      scalars.push_back(b.getNamedAttr("SourceN", i64(types[0].getDimSize(0))));
      scalars.push_back(b.getNamedAttr("N", i64(types[1].getDimSize(0))));
    } else if (spec->kind == "pointwise_loss") {
      allowed.insert("reduction");
      auto reduction = op->getAttrOfType<StringAttr>("reduction");
      if (op->getNumOperands() != 2 || types[0] != types[1] || types[0] != result ||
          !reduction || reduction.getValue() != "none")
        return fail("x86 native pointwise loss requires same-shape f32 operands "
                    "and reduction = \"none\"");
      double parameter = 0.0;
      StringRef parameterName = name == "tessera.loss.huber"       ? "delta"
                                : name == "tessera.loss.smooth_l1" ? "beta"
                                                                   : "";
      if (!parameterName.empty()) {
        allowed.insert(parameterName);
        auto value = op->getAttrOfType<FloatAttr>(parameterName);
        parameter = value ? value.getValueAsDouble() : 1.0;
        if (!std::isfinite(parameter) || parameter <= 0.0)
          return fail("x86 native pointwise loss parameter must be positive finite");
      }
      int64_t n = elements(result);
      if (n <= 0) return fail("x86 native pointwise loss shape exceeds its ABI");
      semantic.push_back(b.getNamedAttr("reduction", reduction));
      scalars.push_back(b.getNamedAttr("N", i64(n)));
      scalars.push_back(b.getNamedAttr("Kind", b.getI32IntegerAttr(lossKind(name))));
      scalars.push_back(b.getNamedAttr("Parameter", b.getF64FloatAttr(parameter)));
    } else {
      bool cholesky = spec->kind == "cholesky";
      allowed.insert("lower");
      if (!cholesky) {
        allowed.insert("trans");
        allowed.insert("unit_diag");
      }
      RankedTensorType matrix = types[0];
      auto lowerAttr = op->getAttrOfType<BoolAttr>("lower");
      if (op->hasAttr("lower") && !lowerAttr)
        return fail("x86 native linalg lower must be a boolean");
      bool lower = !lowerAttr || lowerAttr.getValue();
      for (StringRef flag : {"trans", "unit_diag"})
        if (auto value = op->getAttr(flag)) {
          auto boolean = dyn_cast<BoolAttr>(value);
          if (!boolean || boolean.getValue())
            return fail("x86 native triangular solve implements trans = false, "
                        "unit_diag = false only");
        }
      if (op->getNumOperands() != (cholesky ? 1u : 2u) ||
          (matrix.getRank() != 2 && matrix.getRank() != 3) ||
          matrix.getShape().back() != matrix.getShape()[matrix.getRank() - 2] ||
          llvm::any_of(types, [](RankedTensorType t) {
            return !t.getElementType().isF32();
          }))
        return fail("x86 native linalg requires a square rank-2 or rank-3 f32 matrix");
      int64_t batch = matrix.getRank() == 3 ? matrix.getDimSize(0) : 1;
      int64_t n = matrix.getShape().back();
      scalars.push_back(b.getNamedAttr("Batch", i64(batch)));
      scalars.push_back(b.getNamedAttr("N", i64(n)));
      if (cholesky) {
        // The ABI factors to a lower-triangular L; an upper request is a
        // different result, not a layout hint.
        if (!lower || result != matrix)
          return fail("x86 native cholesky implements lower = true into the "
                      "matrix shape only");
      } else {
        RankedTensorType rhs = types[1];
        if (rhs.getRank() != matrix.getRank() ||
            rhs.getDimSize(rhs.getRank() - 2) != n ||
            (matrix.getRank() == 3 && rhs.getDimSize(0) != batch) || result != rhs)
          return fail("x86 native triangular solve rhs/result disagree with the matrix");
        scalars.push_back(b.getNamedAttr("M", i64(rhs.getShape().back())));
        scalars.push_back(b.getNamedAttr("Lower", b.getI32IntegerAttr(lower ? 1 : 0)));
      }
    }
    semantic.push_back(b.getNamedAttr("abi_key", b.getStringAttr(abi->key)));
  }

  for (NamedAttribute attr : op->getAttrs()) {
    if (!allowed.contains(attr.getName().getValue()))
      return fail("x86 native kernel has an unsupported policy attribute '" +
                  attr.getName().getValue() + "'");
    if (attr.getName() == "tessera.effect_kind" &&
        attr.getValue() != b.getStringAttr("pure"))
      return fail("x86 native kernel requires a pure Graph operation");
  }

  SmallVector<Attribute> shapes, storages;
  for (RankedTensorType ty : types) {
    StringRef storage = x86Storage(ty.getElementType());
    if (storage.empty()) return fail("x86 native kernel storage is unsupported");
    shapes.push_back(b.getDenseI64ArrayAttr(ty.getShape()));
    storages.push_back(b.getStringAttr(storage));
  }
  SmallVector<NamedAttribute> fields{
      b.getNamedAttr("family", b.getStringAttr(family)),
      b.getNamedAttr("kind", b.getStringAttr(spec->kind)),
      b.getNamedAttr("graph_op", b.getStringAttr(spec->op)),
      b.getNamedAttr("bindings", names),
      b.getNamedAttr("shapes", b.getArrayAttr(shapes)),
      b.getNamedAttr("storage", b.getArrayAttr(storages)),
      b.getNamedAttr("layout", b.getStringAttr("row_major")),
      b.getNamedAttr("scalars", b.getDictionaryAttr(scalars)),
  };
  fields.append(semantic.begin(), semantic.end());
  return b.getDictionaryAttr(fields);
}

static bool claimsX86Kernel(ModuleOp mod) {
  auto target = mod->getAttrOfType<StringAttr>("tessera.target");
  return target && target.getValue() == "x86" &&
         mod->hasAttr("tessera.launch_bindings");
}

static LogicalResult scheduleNativeX86Kernel(ModuleOp mod) {
  if (!claimsX86Kernel(mod)) return success();
  SmallVector<Operation *> ops;
  mod.walk([&](Operation *op) { if (x86KernelSpec(op)) ops.push_back(op); });
  for (Operation *op : ops) {
    auto contract = x86KernelContract(op);
    if (failed(contract)) return failure();
    auto fn = op->getParentOfType<func::FuncOp>();
    auto ret = dyn_cast<func::ReturnOp>(fn.getBody().front().back());
    if (fn.getBody().front().getOperations().size() != 2 || !ret ||
        ret.getOperands() != op->getResults())
      return op->emitError("x86 native kernel must return its only operation");
    OpBuilder b(op);
    b.setInsertionPointAfter(op);
    auto hash = b.getStringAttr(x86KernelHash(*contract));
    op->setAttr("schedule.artifact_hash", hash);
    OperationState state(op->getLoc(), "schedule.artifact");
    state.addAttribute("hash", hash);
    state.addAttribute("arch", b.getStringAttr("zen5-avx512"));
    state.addAttribute("shape_key", b.getStringAttr(
        (Twine("family=x86_kernel;kind=") +
         cast<StringAttr>(contract->get("family")).getValue() + "/" +
         cast<StringAttr>(contract->get("kind")).getValue()).str()));
    state.addAttribute("contract", *contract);
    b.create(state);
  }
  return success();
}

static LogicalResult lowerNativeX86Kernel(ModuleOp mod) {
  SmallVector<schedule::ArtifactOp> records;
  mod.walk([&](schedule::ArtifactOp op) {
    if (op.getShapeKey().starts_with("family=x86_kernel;")) records.push_back(op);
  });
  for (auto record : records) {
    auto fn = record->getParentOfType<func::FuncOp>();
    if (!fn || !fn.getBody().hasOneBlock() ||
        fn.getBody().front().getOperations().size() != 3)
      return record.emitError("x86 native kernel Schedule requires its isolated "
                              "Graph parent");
    Operation *op = &fn.getBody().front().front();
    const X86KernelSpec *spec = x86KernelSpec(op);
    if (!spec)
      return record.emitError("x86 native kernel Schedule lost its Graph operation");
    auto contract = x86KernelContract(op);
    auto ret = dyn_cast<func::ReturnOp>(fn.getBody().front().back());
    if (failed(contract) || !ret || ret.getOperands() != op->getResults() ||
        record->getAttr("contract") != *contract ||
        record.getHash() != x86KernelHash(*contract) ||
        record.getArch() != "zen5-avx512" ||
        op->getAttr("schedule.artifact_hash") != record->getAttr("hash"))
      return record.emitError("x86 native kernel Schedule contract was altered");

    MLIRContext *ctx = mod.getContext();
    OpBuilder b(mod.getBody(), mod.getBody()->end());
    Type ptr = LLVM::LLVMPointerType::get(ctx);
    Type i64 = b.getI64Type();
    unsigned inputs = op->getNumOperands();
    StringRef family = spec->family;
    std::string name;
    SmallVector<Type> params;
    const X86AbiSpec *abi = family == "abi" ? x86AbiSpec(spec->subfamily) : nullptr;
    if (abi) {
      name = (Twine("tessera_tile_x86_") + abi->key).str();
      for (StringRef arg : abi->args) {
        SmallVector<StringRef> parts;
        arg.split(parts, ':');
        params.push_back(parts[0] == "b" ? ptr
                         : parts[2] == "i32" ? Type(b.getI32Type())
                         : parts[2] == "f32" ? Type(b.getF32Type())
                                             : i64);
      }
    } else {
      StringRef tileFamily = family == "elementwise" ? spec->subfamily : family;
      name = (Twine("tessera_tile_x86_") + tileFamily + "_" + spec->kind).str();
      params.assign(inputs + 1, ptr);
      params.append(family == "elementwise" ? 1 : 2, i64);
    }
    if (SymbolTable::lookupSymbolIn(mod, name))
      return record.emitError("x86 native kernel symbol already exists");
    auto type = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx), params,
                                            false);
    auto kernel = LLVM::LLVMFuncOp::create(b, op->getLoc(), name, type);
    kernel->setAttr("tessera.x86_kernel_contract", *contract);
    kernel->setAttr("tessera.schedule_hash", record->getAttr("hash"));
    Block *entry = kernel.addEntryBlock(b);
    b.setInsertionPointToStart(entry);
    Location loc = op->getLoc();
    OperationState tile(loc, "tile." + (family == "abi" ? std::string("x86_abi")
                                        : family == "elementwise"
                                            ? std::string("elementwise")
                                            : family.str()) + "_kernel");
    tile.addOperands(entry->getArguments());
    if (family == "elementwise") {
      auto [inStorage, outStorage] = elementwiseStorage(spec->subfamily);
      tile.addAttribute("family", b.getStringAttr(spec->subfamily));
      tile.addAttribute("kind", b.getStringAttr(spec->kind));
      tile.addAttribute("storage", b.getStringAttr(inStorage));
      tile.addAttribute("output_storage", b.getStringAttr(outStorage));
      if (spec->subfamily == "where")
        tile.addAttribute("condition_storage", b.getStringAttr("i8"));
    } else if (family == "argreduce") {
      tile.addAttribute("kind", b.getStringAttr(spec->kind));
      tile.addAttribute("storage", b.getStringAttr("f32"));
      tile.addAttribute("output_storage", b.getStringAttr("i32"));
      tile.addAttribute("tie_break", b.getStringAttr("first"));
    } else if (family == "scan") {
      tile.addAttribute("kind", b.getStringAttr(spec->kind));
      tile.addAttribute("storage", b.getStringAttr("f32"));
      tile.addAttribute("inclusive", b.getBoolAttr(true));
    } else if (family == "rope") {
      tile.addAttribute("storage", b.getStringAttr("f32"));
      tile.addAttribute("layout", b.getStringAttr("interleaved_pairs"));
    } else {
      tile.addAttribute("symbol", b.getStringAttr(abi->symbol));
      tile.addAttribute("abi", b.getStringAttr(abi->abi));
      tile.addAttribute("family", b.getStringAttr(abi->family));
      tile.addAttribute("effects", b.getStringAttr(abi->effects));
      tile.addAttribute("returns_status", b.getBoolAttr(false));
    }
    b.create(tile);
    LLVM::ReturnOp::create(b, loc, ValueRange{});
    fn.erase();
  }
  return success();
}

} // namespace
