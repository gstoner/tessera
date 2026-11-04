// E2E-REAL-6, Apple CPU batched GEMM + LAPACK factorizations (2026-11-04).
//
// One isolated static Apple CPU Graph operation -> a content-addressed
// Schedule record -> the registered Tile value op the Apple CPU value lowering
// (`tile-to-apple_cpu{value=true}`) turns into a verified
// `tessera_apple.cpu.call`. This is the native owner of what the Graph-owned
// branch of `apple_cpu_native.package_native` used to decide in Python for
// `batched_gemm`, `tri_solve`, `cholesky`, `cholesky_solve`, `lu`, `qr` and
// `svd`; the retired constructor survives only as the declared differential
// oracle (`tests/_support/apple_cpu_baseline.py`, Decision #31(a)).
//
// The consumer of every family is a hand-written Accelerate/LAPACK C ABI entry
// in `apple_cpu_runtime.cpp` -- a Decision #28 Tier-3 delegate reached through
// the declared Target IR op, one lowering path per Decision #31. LAPACK owns
// its blocking, so the Schedule records only the semantic and ABI contract.
//
// Ownership is opt-in per module: the Graph must name
// `tessera.target = "apple_cpu"`, `tessera.arch = "apple_silicon_cpu"` and
// carry `tessera.launch_bindings` (the host buffer aliases, inputs then
// results). Without the bindings attribute these ops pass Graph -> Schedule
// untouched; with it, anything outside the envelope below fails closed
// (Decision #21a): unknown attributes, non-static shapes, the wrong element
// types, a semantic flag the C ABI does not implement.
namespace {

struct AppleCPUKernelSpec {
  StringRef op;      // Graph op name
  StringRef kind;    // Tile op suffix and value-call op_kind
  StringRef symbol;  // the C ABI entry (TileToApple.cpp kLinalgSpecs agree)
  StringRef abi;     // the delegated library routine
  unsigned inputs;
  unsigned results;
};

// Must agree with `python/tessera/compiler/native_apple_cpu_kernel.py`
// (`APPLE_CPU_KERNEL_OPS`) and with the Apple CPU value lowering's symbols;
// `tests/unit/test_apple_cpu_kernel_migration.py` pins both.
static const AppleCPUKernelSpec kAppleCPUKernelSpecs[] = {
    {"tessera.batched_gemm", "batched_gemm", "tessera_apple_cpu_gemm_f32_batched",
     "cblas_sgemm_batched_loop", 2, 1},
    {"tessera.tri_solve", "tri_solve", "tessera_apple_cpu_tri_solve_f32",
     "lapack_strtrs", 2, 1},
    {"tessera.cholesky", "cholesky", "tessera_apple_cpu_cholesky_f32",
     "lapack_spotrf", 1, 1},
    {"tessera.cholesky_solve", "cholesky_solve",
     "tessera_apple_cpu_cholesky_solve_f32", "lapack_spotrs", 2, 1},
    {"tessera.lu", "lu", "tessera_apple_cpu_lu_f32", "lapack_sgetrf", 1, 2},
    {"tessera.qr", "qr", "tessera_apple_cpu_qr_f32", "lapack_sgeqrf", 1, 2},
    {"tessera.svd", "svd", "tessera_apple_cpu_svd_f32", "lapack_sgesvd", 1, 3},
};

static const AppleCPUKernelSpec *appleCPUKernelSpec(Operation *op) {
  StringRef name = op->getName().getStringRef();
  for (const AppleCPUKernelSpec &spec : kAppleCPUKernelSpecs)
    if (spec.op == name)
      return &spec;
  return nullptr;
}

static std::string appleCPUKernelHash(DictionaryAttr contract) {
  std::string text;
  llvm::raw_string_ostream os(text);
  contract.print(os);
  os.flush();
  return llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(text)), true);
}

// Derive the contract for one owned Graph op, or fail with a diagnostic. The
// same function re-derives it at Schedule -> Tile, so a Schedule record that
// no longer matches its Graph parent is refused.
static FailureOr<DictionaryAttr> appleCPUKernelContract(Operation *op) {
  const AppleCPUKernelSpec *spec = appleCPUKernelSpec(op);
  auto fn = op->getParentOfType<func::FuncOp>();
  auto mod = op->getParentOfType<ModuleOp>();
  auto fail = [&](const Twine &message) -> FailureOr<DictionaryAttr> {
    op->emitError("Apple CPU native kernel: " + message);
    return failure();
  };
  if (!spec || !fn || !mod)
    return fail("requires a registered owned op");
  auto target = mod->getAttrOfType<StringAttr>("tessera.target");
  auto arch = mod->getAttrOfType<StringAttr>("tessera.arch");
  if (!target || target.getValue() != "apple_cpu" || !arch ||
      arch.getValue() != "apple_silicon_cpu")
    return fail("requires the apple_silicon_cpu Apple CPU target");
  if (!fn.getBody().hasOneBlock() || op->getNumOperands() != spec->inputs ||
      op->getNumResults() != spec->results ||
      fn.getNumArguments() != spec->inputs ||
      fn.getNumResults() != spec->results)
    return fail("requires an isolated entry whose arguments are its operands "
                "and whose results are its results");
  // Each operand is a distinct entry argument: a launch descriptor names each
  // buffer once, and the ABI reads them as separate inputs.
  llvm::SmallDenseSet<unsigned> used;
  for (Value operand : op->getOperands()) {
    auto arg = dyn_cast<BlockArgument>(operand);
    if (!arg || arg.getOwner() != &fn.getBody().front() ||
        !used.insert(arg.getArgNumber()).second)
      return fail("operands must be distinct entry arguments");
  }
  for (unsigned index = 0; index < fn.getNumArguments(); ++index) {
    auto ty = dyn_cast<RankedTensorType>(fn.getArgument(index).getType());
    if (auto attrs = fn.getArgAttrDict(index))
      for (NamedAttribute attr : attrs) {
        if (attr.getName() == "tessera.layout") {
          auto layout = dyn_cast<StringAttr>(attr.getValue());
          if (!layout || layout.getValue() != "row_major")
            return fail("requires row_major arguments");
          continue;
        }
        auto names = dyn_cast<ArrayAttr>(attr.getValue());
        if (attr.getName() != "tessera.dim_names" || !names || !ty ||
            names.size() != static_cast<size_t>(ty.getRank()))
          return fail("argument policy is unsupported");
      }
  }

  // Every extent is a C `int` in the ABI.
  constexpr int64_t kInt32Max = std::numeric_limits<int32_t>::max();
  SmallVector<RankedTensorType> types;
  auto addType = [&](Type type) {
    auto ty = dyn_cast<RankedTensorType>(type);
    if (!ty || !ty.hasStaticShape() || ty.getEncoding() ||
        llvm::any_of(ty.getShape(),
                     [&](int64_t d) { return d <= 0 || d > kInt32Max; }))
      return false;
    types.push_back(ty);
    return true;
  };
  for (Value value : op->getOperands())
    if (!addType(value.getType()))
      return fail("requires static positive int32-extent operands");
  for (auto [index, value] : llvm::enumerate(op->getResults()))
    if (!addType(value.getType()) || fn.getResultTypes()[index] != value.getType())
      return fail("requires static positive int32-extent results");

  auto names = mod->getAttrOfType<ArrayAttr>("tessera.launch_bindings");
  if (!names || names.size() != types.size())
    return fail("requires one binding name per operand and result");
  llvm::StringSet<> seen;
  for (Attribute attr : names) {
    auto name = dyn_cast<StringAttr>(attr);
    if (!name || name.getValue().empty() || !seen.insert(name.getValue()).second)
      return fail("binding names must be distinct and non-empty");
  }

  OpBuilder b(op);
  auto i64 = [&](int64_t v) { return b.getI64IntegerAttr(v); };
  auto f32 = [](RankedTensorType ty) { return ty.getElementType().isF32(); };
  auto shapeIs = [](RankedTensorType ty, ArrayRef<int64_t> shape) {
    return ty.getShape() == shape;
  };
  SmallVector<NamedAttribute> scalars;    // the ABI's integer arguments
  SmallVector<NamedAttribute> semantics;  // the materialized semantic flags
  llvm::StringSet<> allowed{"tessera.effect_kind", "schedule.artifact_hash"};
  // A semantic flag that selects a different result is materialized, never
  // defaulted silently (Decision #21a): read it, check the ABI implements it,
  // and record the value the delegate will compute.
  auto flag = [&](StringRef name, bool fallback, bool &out) {
    allowed.insert(name);
    Attribute value = op->getAttr(name);
    auto boolean = dyn_cast_or_null<BoolAttr>(value);
    if (value && !boolean) return false;
    out = boolean ? boolean.getValue() : fallback;
    semantics.push_back(b.getNamedAttr(name, b.getBoolAttr(out)));
    return true;
  };

  StringRef kind = spec->kind;
  if (kind == "batched_gemm") {
    RankedTensorType lhs = types[0], rhs = types[1], out = types[2];
    if (lhs.getRank() != 3 || rhs.getRank() != 3 || !f32(lhs) || !f32(rhs) ||
        !f32(out))
      return fail("batched_gemm requires rank-3 f32 operands and result");
    int64_t batch = lhs.getDimSize(0), m = lhs.getDimSize(1),
            k = lhs.getDimSize(2), n = rhs.getDimSize(2);
    if (!shapeIs(rhs, {batch, k, n}) || !shapeIs(out, {batch, m, n}))
      return fail("batched_gemm requires [B,M,K] x [B,K,N] -> [B,M,N] with no "
                  "broadcasting");
    // The ABI's per-batch strides are C `int` element counts.
    if (m * k > kInt32Max || k * n > kInt32Max || m * n > kInt32Max)
      return fail("batched_gemm per-batch stride exceeds the int32 ABI");
    scalars.append({b.getNamedAttr("Batch", i64(batch)), b.getNamedAttr("M", i64(m)),
                    b.getNamedAttr("N", i64(n)), b.getNamedAttr("K", i64(k)),
                    b.getNamedAttr("StrideA", i64(m * k)),
                    b.getNamedAttr("StrideB", i64(k * n)),
                    b.getNamedAttr("StrideC", i64(m * n))});
  } else if (kind == "tri_solve" || kind == "cholesky_solve" ||
             kind == "cholesky") {
    RankedTensorType matrix = types[0];
    if (matrix.getRank() != 2 || matrix.getDimSize(0) != matrix.getDimSize(1) ||
        llvm::any_of(types, [&](RankedTensorType t) { return !f32(t); }))
      return fail(kind + " requires a square rank-2 f32 matrix");
    int64_t n = matrix.getDimSize(0);
    bool lower = true;
    if (!flag("lower", true, lower))
      return fail(kind + " lower must be a boolean");
    scalars.push_back(b.getNamedAttr("N", i64(n)));
    if (kind == "cholesky") {
      // spotrf on the row-major buffer writes the lower factor only; an upper
      // request is a different result, not a layout hint.
      if (!lower || types[1] != matrix)
        return fail("cholesky implements lower = true into the matrix shape only");
    } else {
      RankedTensorType rhs = types[1], out = types[2];
      if (rhs.getRank() != 2 || rhs.getDimSize(0) != n || out != rhs)
        return fail(kind + " requires rhs [N,K] and a result of the rhs shape");
      scalars.push_back(b.getNamedAttr("K", i64(rhs.getDimSize(1))));
      if (kind == "tri_solve") {
        bool trans = false, unitDiag = false;
        if (!flag("trans", false, trans) || !flag("unit_diag", false, unitDiag))
          return fail("tri_solve trans/unit_diag must be booleans");
      }
    }
  } else if (kind == "lu") {
    RankedTensorType matrix = types[0];
    int64_t n = matrix.getDimSize(0);
    if (matrix.getRank() != 2 || matrix.getDimSize(1) != n || !f32(matrix) ||
        types[1] != matrix || !shapeIs(types[2], {n}) ||
        !types[2].getElementType().isInteger(32))
      return fail("lu requires a square f32 [N,N] -> ([N,N] f32, [N] i32)");
    scalars.push_back(b.getNamedAttr("N", i64(n)));
  } else {  // qr | svd
    RankedTensorType matrix = types[0];
    if (matrix.getRank() != 2 ||
        llvm::any_of(types, [&](RankedTensorType t) { return !f32(t); }))
      return fail(kind + " requires an f32 rank-2 matrix and f32 results");
    int64_t m = matrix.getDimSize(0), n = matrix.getDimSize(1);
    bool full = false;
    if (!flag("full_matrices", false, full))
      return fail(kind + " full_matrices must be a boolean");
    // Both ABIs compute the reduced factorization only.
    if (full)
      return fail(kind + " implements full_matrices = false only");
    if (kind == "qr") {
      if (m < n || !shapeIs(types[1], {m, n}) || !shapeIs(types[2], {n, n}))
        return fail("qr requires M >= N and reduced (Q [M,N], R [N,N])");
    } else {
      int64_t r = std::min(m, n);
      if (!shapeIs(types[1], {m, r}) || !shapeIs(types[2], {r}) ||
          !shapeIs(types[3], {r, n}))
        return fail("svd requires reduced (U [M,K], S [K], V [K,N]), K = min(M,N)");
    }
    scalars.append({b.getNamedAttr("M", i64(m)), b.getNamedAttr("N", i64(n))});
  }

  for (NamedAttribute attr : op->getAttrs()) {
    if (!allowed.contains(attr.getName().getValue()))
      return fail("has an unsupported policy attribute '" +
                  attr.getName().getValue() + "'");
    if (attr.getName() == "tessera.effect_kind" &&
        attr.getValue() != b.getStringAttr("pure"))
      return fail("requires a pure Graph operation");
  }

  SmallVector<Attribute> shapes, storages;
  for (RankedTensorType ty : types) {
    shapes.push_back(b.getDenseI64ArrayAttr(ty.getShape()));
    storages.push_back(b.getStringAttr(ty.getElementType().isF32() ? "f32" : "i32"));
  }
  return b.getDictionaryAttr({
      b.getNamedAttr("kind", b.getStringAttr(kind)),
      b.getNamedAttr("graph_op", b.getStringAttr(spec->op)),
      b.getNamedAttr("symbol", b.getStringAttr(spec->symbol)),
      b.getNamedAttr("abi", b.getStringAttr(spec->abi)),
      b.getNamedAttr("framework", b.getStringAttr("Accelerate")),
      b.getNamedAttr("inputs", i64(spec->inputs)),
      b.getNamedAttr("bindings", names),
      b.getNamedAttr("shapes", b.getArrayAttr(shapes)),
      b.getNamedAttr("storage", b.getArrayAttr(storages)),
      b.getNamedAttr("layout", b.getStringAttr("row_major")),
      b.getNamedAttr("scalars", b.getDictionaryAttr(scalars)),
      b.getNamedAttr("semantics", b.getDictionaryAttr(semantics)),
  });
}

static bool claimsAppleCPUKernel(ModuleOp mod) {
  auto target = mod->getAttrOfType<StringAttr>("tessera.target");
  return target && target.getValue() == "apple_cpu" &&
         mod->hasAttr("tessera.launch_bindings");
}

static LogicalResult scheduleNativeAppleCPUKernel(ModuleOp mod, bool &selected) {
  selected = false;
  if (!claimsAppleCPUKernel(mod)) return success();
  SmallVector<Operation *> ops;
  mod.walk([&](Operation *op) { if (appleCPUKernelSpec(op)) ops.push_back(op); });
  // A bindings-carrying module with none of these ops belongs to another
  // contract; one with several is no isolated request this ABI can serve.
  if (ops.empty()) return success();
  if (ops.size() != 1)
    return mod.emitError("Apple CPU native kernel requires exactly one owned "
                         "operation in a module carrying launch bindings");
  Operation *op = ops.front();
  auto contract = appleCPUKernelContract(op);
  if (failed(contract)) return failure();
  auto fn = op->getParentOfType<func::FuncOp>();
  auto ret = dyn_cast<func::ReturnOp>(fn.getBody().front().back());
  if (fn.getBody().front().getOperations().size() != 2 || !ret ||
      ret.getOperands() != op->getResults())
    return op->emitError("Apple CPU native kernel must return its only "
                         "operation's results in order");
  OpBuilder b(op);
  b.setInsertionPointAfter(op);
  auto hash = b.getStringAttr(appleCPUKernelHash(*contract));
  op->setAttr("schedule.artifact_hash", hash);
  OperationState state(op->getLoc(), "schedule.artifact");
  state.addAttribute("hash", hash);
  state.addAttribute("arch", b.getStringAttr("apple_silicon_cpu"));
  state.addAttribute("shape_key", b.getStringAttr(
      (Twine("family=apple_cpu_kernel;kind=") +
       cast<StringAttr>(contract->get("kind")).getValue()).str()));
  state.addAttribute("contract", *contract);
  b.create(state);
  selected = true;
  return success();
}

static LogicalResult lowerNativeAppleCPUKernel(ModuleOp mod, bool &selected) {
  selected = false;
  SmallVector<schedule::ArtifactOp> records;
  mod.walk([&](schedule::ArtifactOp op) {
    if (op.getShapeKey().starts_with("family=apple_cpu_kernel;")) records.push_back(op);
  });
  if (records.empty()) return success();
  if (records.size() != 1)
    return mod.emitError("Apple CPU native kernel Schedule requires one record");
  auto record = records.front();
  auto fn = record->getParentOfType<func::FuncOp>();
  if (!fn || !fn.getBody().hasOneBlock() ||
      fn.getBody().front().getOperations().size() != 3)
    return record.emitError("Apple CPU native kernel Schedule requires its "
                            "isolated Graph parent");
  Operation *op = &fn.getBody().front().front();
  const AppleCPUKernelSpec *spec = appleCPUKernelSpec(op);
  if (!spec)
    return record.emitError("Apple CPU native kernel Schedule lost its Graph operation");
  auto contract = appleCPUKernelContract(op);
  auto ret = dyn_cast<func::ReturnOp>(fn.getBody().front().back());
  if (failed(contract) || !ret || ret.getOperands() != op->getResults() ||
      record->getAttr("contract") != *contract ||
      record.getHash() != appleCPUKernelHash(*contract) ||
      record.getArch() != "apple_silicon_cpu" ||
      op->getAttr("schedule.artifact_hash") != record->getAttr("hash") ||
      record.getShapeKey() != (Twine("family=apple_cpu_kernel;kind=") + spec->kind).str())
    return record.emitError("Apple CPU native kernel Schedule contract was altered");

  // The Tile value op the Apple CPU value lowering consumes. Semantic flags
  // are written explicitly from the contract, so the Target op never reads a
  // default the Graph op left implicit.
  OpBuilder b(op);
  OperationState tile(op->getLoc(), ("tile." + spec->kind).str());
  tile.addOperands(op->getOperands());
  tile.addTypes(op->getResultTypes());
  tile.addAttribute("source", b.getStringAttr(spec->op));
  for (NamedAttribute attr : cast<DictionaryAttr>(contract->get("semantics")))
    tile.addAttribute(attr.getName(), attr.getValue());
  tile.addAttribute("tessera.apple_cpu_kernel_contract", *contract);
  tile.addAttribute("tessera.schedule_hash", record->getAttr("hash"));
  Operation *lowered = b.create(tile);
  op->replaceAllUsesWith(lowered->getResults());
  record.erase();
  op->erase();
  selected = true;
  return success();
}

} // namespace
