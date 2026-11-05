// E2E-REAL-6, Apple GPU linalg factorizations (2026-11-04).
//
// One isolated static Apple GPU Graph operation -> a content-addressed
// Schedule record -> the registered Tile value op the Apple GPU value lowering
// (`tile-to-apple_gpu{value=true}`) turns into a verified
// `tessera_apple.gpu.kernel_call`. This is the native owner of what
// `apple_native.package_svd` and the generic value branch of
// `apple_native.package_native` used to decide in Python for `svd`,
// `cholesky`, `cholesky_solve` and `tri_solve`; those constructors survive
// only as the declared differential oracle (Decision #31(a)). The Apple CPU
// sibling is `NativeAppleCPUKernel.h`; the envelopes differ (the Metal kernels
// take a rank-3 batch of matrices, and the Jacobi SVD factors tall matrices
// only), so the contracts are separate.
//
// The consumer of every family is a hand-written MSL / MPS C ABI entry in
// `apple_gpu_runtime.mm` -- a Decision #28 Tier-3 delegate reached through the
// declared Target IR op, one lowering path per Decision #31.
//
// Ownership is opt-in per module: the Graph must name
// `tessera.target = "apple_gpu"`, `tessera.arch = "apple_gpu"` and carry
// `tessera.launch_bindings` (the buffer aliases, inputs then results). Without
// the bindings attribute these ops pass Graph -> Schedule untouched; with it,
// anything outside the envelope below fails closed (Decision #21a): unknown
// attributes, non-static shapes, non-f32 storage, a semantic flag the Metal
// kernel does not implement.
namespace {

struct AppleGPUKernelSpec {
  StringRef op;    // Graph op name
  StringRef kind;  // Tile op suffix and value-call op_kind
  unsigned inputs;
  unsigned results;
};

// Must agree with `python/tessera/compiler/scheduled_apple_gpu_kernel.py`
// (`APPLE_GPU_KERNEL_OPS`); the C ABI symbol and ABI id are chosen by the
// Apple GPU value lowering (`nativeAppleGPUKernelEntry` in TileToApple.cpp),
// which refuses a contract whose recorded symbol disagrees with its own.
// `tests/unit/test_apple_gpu_kernel_migration.py` pins all three.
static const AppleGPUKernelSpec kAppleGPUKernelSpecs[] = {
    {"tessera.cholesky", "cholesky", 1, 1},
    {"tessera.tri_solve", "tri_solve", 2, 1},
    {"tessera.cholesky_solve", "cholesky_solve", 2, 1},
    {"tessera.svd", "svd", 1, 3},
};

static const AppleGPUKernelSpec *appleGPUKernelSpec(Operation *op) {
  StringRef name = op->getName().getStringRef();
  for (const AppleGPUKernelSpec &spec : kAppleGPUKernelSpecs)
    if (spec.op == name)
      return &spec;
  return nullptr;
}

static std::string appleGPUKernelHash(DictionaryAttr contract) {
  std::string text;
  llvm::raw_string_ostream os(text);
  contract.print(os);
  os.flush();
  return llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(text)), true);
}

// The C ABI entry and ABI id for one admitted request. Mirrors the value
// lowering's table; recorded in the contract so the replayed artifact names
// what the lowering must produce.
static std::pair<StringRef, StringRef> appleGPUKernelEntry(StringRef kind,
                                                           bool batched) {
  if (kind == "cholesky")
    return {"tessera_apple_gpu_cholesky_f32", "tessera.apple.value.cholesky.f32.v1"};
  if (kind == "tri_solve")
    return {"tessera_apple_gpu_tri_solve_f32", "tessera.apple.value.tri_solve.f32.v1"};
  if (kind == "cholesky_solve")
    return {"tessera_apple_gpu_solve_cholesky_f32",
            "tessera.apple.value.cholesky_solve.f32.v1"};
  if (batched)
    return {"tessera_apple_gpu_svd_reduced_batched_f32",
            "tessera.apple.svd.a_u_s_vh_batch_m_n.f32.v1"};
  return {"tessera_apple_gpu_svd_reduced_f32", "tessera.apple.svd.a_u_s_vh_m_n.f32.v1"};
}

// Derive the contract for one owned Graph op, or fail with a diagnostic. The
// same function re-derives it at Schedule -> Tile, so a Schedule record that
// no longer matches its Graph parent is refused.
static FailureOr<DictionaryAttr> appleGPUKernelContract(Operation *op) {
  const AppleGPUKernelSpec *spec = appleGPUKernelSpec(op);
  auto fn = op->getParentOfType<func::FuncOp>();
  auto mod = op->getParentOfType<ModuleOp>();
  auto fail = [&](const Twine &message) -> FailureOr<DictionaryAttr> {
    op->emitError("Apple GPU native kernel: " + message);
    return failure();
  };
  if (!spec || !fn || !mod)
    return fail("requires a registered owned op");
  auto target = mod->getAttrOfType<StringAttr>("tessera.target");
  auto arch = mod->getAttrOfType<StringAttr>("tessera.arch");
  if (!target || target.getValue() != "apple_gpu" || !arch ||
      arch.getValue() != "apple_gpu")
    return fail("requires the apple_gpu Apple GPU target");
  if (!fn.getBody().hasOneBlock() || op->getNumOperands() != spec->inputs ||
      op->getNumResults() != spec->results ||
      fn.getNumArguments() != spec->inputs ||
      fn.getNumResults() != spec->results)
    return fail("requires an isolated entry whose arguments are its operands "
                "and whose results are its results");
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

  // Every extent, and every flattened batch, is a C `int32_t` in the ABI.
  constexpr int64_t kInt32Max = std::numeric_limits<int32_t>::max();
  SmallVector<RankedTensorType> types;
  auto addType = [&](Type type) {
    auto ty = dyn_cast<RankedTensorType>(type);
    if (!ty || !ty.hasStaticShape() || ty.getEncoding() ||
        !ty.getElementType().isF32() || ty.getRank() == 0 ||
        ty.getNumElements() > kInt32Max ||
        llvm::any_of(ty.getShape(), [](int64_t d) { return d <= 0; }))
      return false;
    types.push_back(ty);
    return true;
  };
  for (Value value : op->getOperands())
    if (!addType(value.getType()))
      return fail("requires static positive f32 operands within the int32 ABI");
  for (auto [index, value] : llvm::enumerate(op->getResults()))
    if (!addType(value.getType()) || fn.getResultTypes()[index] != value.getType())
      return fail("requires static positive f32 results within the int32 ABI");

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
  SmallVector<NamedAttribute> scalars;    // the ABI's integer extents
  SmallVector<NamedAttribute> semantics;  // the materialized semantic flags
  llvm::StringSet<> allowed{"tessera.effect_kind", "schedule.artifact_hash"};
  // A semantic flag that selects a different result is materialized, never
  // defaulted silently (Decision #21a).
  auto flag = [&](StringRef name, bool fallback, bool &out) {
    allowed.insert(name);
    Attribute value = op->getAttr(name);
    auto boolean = dyn_cast_or_null<BoolAttr>(value);
    if (value && !boolean) return false;
    out = boolean ? boolean.getValue() : fallback;
    semantics.push_back(b.getNamedAttr(name, b.getBoolAttr(out)));
    return true;
  };
  // Leading dimensions are a batch of independent matrices.
  auto batchOf = [](RankedTensorType ty, int64_t trailing) {
    int64_t batch = 1;
    for (int64_t d : ty.getShape().drop_back(trailing)) batch *= d;
    return batch;
  };

  StringRef kind = spec->kind;
  bool batched = false;
  if (kind == "svd") {
    RankedTensorType matrix = types[0];
    int64_t rank = matrix.getRank();
    if (rank != 2 && rank != 3)
      return fail("svd requires a rank-2 matrix or a rank-3 batch of matrices");
    bool full = false;
    if (!flag("full_matrices", false, full))
      return fail("svd full_matrices must be a boolean");
    // The reduced Jacobi adapter computes the reduced factorization only.
    if (full)
      return fail("svd implements full_matrices = false only");
    ArrayRef<int64_t> lead = matrix.getShape().drop_back(2);
    int64_t m = matrix.getDimSize(rank - 2), n = matrix.getDimSize(rank - 1);
    // The one-sided Jacobi kernel factors tall matrices; a wide request is a
    // different kernel, not a layout hint.
    if (m < n)
      return fail("svd requires a tall matrix (M >= N)");
    auto with = [&](ArrayRef<int64_t> tail) {
      SmallVector<int64_t> shape(lead.begin(), lead.end());
      shape.append(tail.begin(), tail.end());
      return shape;
    };
    if (types[1].getShape() != ArrayRef<int64_t>(with({m, n})) ||
        types[2].getShape() != ArrayRef<int64_t>(with({n})) ||
        types[3].getShape() != ArrayRef<int64_t>(with({n, n})))
      return fail("svd requires reduced (U [..,M,N], S [..,N], Vh [..,N,N])");
    batched = rank == 3;
    if (batched) scalars.push_back(b.getNamedAttr("Batch", i64(lead.front())));
    scalars.append({b.getNamedAttr("M", i64(m)), b.getNamedAttr("N", i64(n))});
  } else {
    RankedTensorType matrix = types[0];
    int64_t rank = matrix.getRank();
    if ((rank != 2 && rank != 3) ||
        matrix.getDimSize(rank - 1) != matrix.getDimSize(rank - 2))
      return fail(kind + " requires a square [N,N] or [B,N,N] matrix");
    int64_t n = matrix.getDimSize(rank - 1);
    bool lower = true;
    if (!flag("lower", true, lower))
      return fail(kind + " lower must be a boolean");
    batched = rank > 2;
    if (kind == "cholesky") {
      // The Metal kernels write the lower factor only; an upper request is a
      // different result, not a layout hint.
      if (!lower || types[1] != matrix)
        return fail("cholesky implements lower = true into the matrix shape only");
      scalars.append({b.getNamedAttr("Batch", i64(batchOf(matrix, 2))),
                      b.getNamedAttr("N", i64(n))});
    } else {
      // The right-hand side is [N,K] (or [B,N,K] beside a batch of matrices);
      // the result has its shape.
      RankedTensorType rhs = types[1], out = types[2];
      if (rhs.getRank() != rank || rhs.getDimSize(rank - 2) != n ||
          (rank == 3 && rhs.getDimSize(0) != matrix.getDimSize(0)) || out != rhs)
        return fail(kind + " requires rhs [..,N,K] with the matrix's batch and a "
                           "result of the rhs shape");
      if (kind == "cholesky_solve" && !lower)
        return fail("cholesky_solve implements lower = true only");
      if (kind == "tri_solve") {
        bool trans = false, unitDiag = false;
        if (!flag("trans", false, trans) || !flag("unit_diag", false, unitDiag))
          return fail("tri_solve trans/unit_diag must be booleans");
      }
      scalars.append({b.getNamedAttr("Batch", i64(batchOf(matrix, 2))),
                      b.getNamedAttr("N", i64(n)),
                      b.getNamedAttr("K", i64(rhs.getDimSize(rank - 1)))});
    }
  }

  for (NamedAttribute attr : op->getAttrs()) {
    if (!allowed.contains(attr.getName().getValue()))
      return fail("has an unsupported policy attribute '" +
                  attr.getName().getValue() + "'");
    if (attr.getName() == "tessera.effect_kind" &&
        attr.getValue() != b.getStringAttr("pure"))
      return fail("requires a pure Graph operation");
  }

  auto [symbol, abi] = appleGPUKernelEntry(kind, batched);
  SmallVector<Attribute> shapes, storages;
  for (RankedTensorType ty : types) {
    shapes.push_back(b.getDenseI64ArrayAttr(ty.getShape()));
    storages.push_back(b.getStringAttr("f32"));
  }
  return b.getDictionaryAttr({
      b.getNamedAttr("kind", b.getStringAttr(kind)),
      b.getNamedAttr("graph_op", b.getStringAttr(spec->op)),
      b.getNamedAttr("symbol", b.getStringAttr(symbol)),
      b.getNamedAttr("abi", b.getStringAttr(abi)),
      b.getNamedAttr("framework", b.getStringAttr("Metal")),
      b.getNamedAttr("inputs", i64(spec->inputs)),
      b.getNamedAttr("bindings", names),
      b.getNamedAttr("shapes", b.getArrayAttr(shapes)),
      b.getNamedAttr("storage", b.getArrayAttr(storages)),
      b.getNamedAttr("layout", b.getStringAttr("row_major")),
      b.getNamedAttr("scalars", b.getDictionaryAttr(scalars)),
      b.getNamedAttr("semantics", b.getDictionaryAttr(semantics)),
  });
}

static bool claimsAppleGPUKernel(ModuleOp mod) {
  auto target = mod->getAttrOfType<StringAttr>("tessera.target");
  return target && target.getValue() == "apple_gpu" &&
         mod->hasAttr("tessera.launch_bindings");
}

static LogicalResult scheduleNativeAppleGPUKernel(ModuleOp mod, bool &selected) {
  selected = false;
  if (!claimsAppleGPUKernel(mod)) return success();
  SmallVector<Operation *> ops;
  mod.walk([&](Operation *op) { if (appleGPUKernelSpec(op)) ops.push_back(op); });
  // A bindings-carrying module with none of these ops belongs to another
  // contract; one with several is no isolated request this ABI can serve.
  if (ops.empty()) return success();
  if (ops.size() != 1)
    return mod.emitError("Apple GPU native kernel requires exactly one owned "
                         "operation in a module carrying launch bindings");
  Operation *op = ops.front();
  auto contract = appleGPUKernelContract(op);
  if (failed(contract)) return failure();
  auto fn = op->getParentOfType<func::FuncOp>();
  auto ret = dyn_cast<func::ReturnOp>(fn.getBody().front().back());
  if (fn.getBody().front().getOperations().size() != 2 || !ret ||
      ret.getOperands() != op->getResults())
    return op->emitError("Apple GPU native kernel must return its only "
                         "operation's results in order");
  OpBuilder b(op);
  b.setInsertionPointAfter(op);
  auto hash = b.getStringAttr(appleGPUKernelHash(*contract));
  op->setAttr("schedule.artifact_hash", hash);
  OperationState state(op->getLoc(), "schedule.artifact");
  state.addAttribute("hash", hash);
  state.addAttribute("arch", b.getStringAttr("apple_gpu"));
  state.addAttribute("shape_key", b.getStringAttr(
      (Twine("family=apple_gpu_kernel;kind=") +
       cast<StringAttr>(contract->get("kind")).getValue()).str()));
  state.addAttribute("contract", *contract);
  b.create(state);
  selected = true;
  return success();
}

static LogicalResult lowerNativeAppleGPUKernel(ModuleOp mod, bool &selected) {
  selected = false;
  SmallVector<schedule::ArtifactOp> records;
  mod.walk([&](schedule::ArtifactOp op) {
    if (op.getShapeKey().starts_with("family=apple_gpu_kernel;")) records.push_back(op);
  });
  if (records.empty()) return success();
  if (records.size() != 1)
    return mod.emitError("Apple GPU native kernel Schedule requires one record");
  auto record = records.front();
  auto fn = record->getParentOfType<func::FuncOp>();
  if (!fn || !fn.getBody().hasOneBlock() ||
      fn.getBody().front().getOperations().size() != 3)
    return record.emitError("Apple GPU native kernel Schedule requires its "
                            "isolated Graph parent");
  Operation *op = &fn.getBody().front().front();
  const AppleGPUKernelSpec *spec = appleGPUKernelSpec(op);
  if (!spec)
    return record.emitError("Apple GPU native kernel Schedule lost its Graph operation");
  auto contract = appleGPUKernelContract(op);
  auto ret = dyn_cast<func::ReturnOp>(fn.getBody().front().back());
  if (failed(contract) || !ret || ret.getOperands() != op->getResults() ||
      record->getAttr("contract") != *contract ||
      record.getHash() != appleGPUKernelHash(*contract) ||
      record.getArch() != "apple_gpu" ||
      op->getAttr("schedule.artifact_hash") != record->getAttr("hash") ||
      record.getShapeKey() != (Twine("family=apple_gpu_kernel;kind=") + spec->kind).str())
    return record.emitError("Apple GPU native kernel Schedule contract was altered");

  // The Tile value op the Apple GPU value lowering consumes. Semantic flags
  // are written explicitly from the contract, so the Target op never reads a
  // default the Graph op left implicit.
  OpBuilder b(op);
  OperationState tile(op->getLoc(), ("tile." + spec->kind).str());
  tile.addOperands(op->getOperands());
  tile.addTypes(op->getResultTypes());
  tile.addAttribute("source", b.getStringAttr(spec->op));
  for (NamedAttribute attr : cast<DictionaryAttr>(contract->get("semantics")))
    tile.addAttribute(attr.getName(), attr.getValue());
  tile.addAttribute("tessera.apple_gpu_kernel_contract", *contract);
  tile.addAttribute("tessera.schedule_hash", record->getAttr("hash"));
  Operation *lowered = b.create(tile);
  op->replaceAllUsesWith(lowered->getResults());
  record.erase();
  op->erase();
  selected = true;
  return success();
}

} // namespace
