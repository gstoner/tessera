// E2E-REAL-6, Apple GPU runtime-descriptor families (2026-11-04).
//
// One isolated Apple GPU Graph operation -> a content-addressed Schedule
// record -> a Tile value op that the Apple GPU value lowering
// (`tile-to-apple_gpu{value=true}`) turns into a verified
// `tessera_apple.gpu.kernel_call`. This is the native owner of what
// `apple_native.package_{batched_gemm,transpose,dynamic_softmax,dynamic_gelu,
// dynamic_popcount,dynamic_count_nonzero,dynamic_topk}` used to decide in
// Python from the Graph object; those constructors survive only as the
// declared differential oracle (Decision #31(a)).
//
// The consumer of every family is a hand-written MPS / MPSGraph / MSL entry in
// `apple_gpu_runtime.mm` -- a Decision #28 Tier-3 delegate reached through the
// declared Target IR op, one lowering path per Decision #31. The runtime owns
// its own blocking, so the Schedule records only the semantic and ABI
// contract: storage, shapes (-1 = a dynamic extent bound by an ABI scalar),
// the ABI scalars the launcher must supply, and every semantic choice the
// kernel makes (permutation, axis, k, ordering policy).
//
// Ownership is opt-in per module: the Graph must name
// `tessera.target = "apple_gpu"`, `tessera.arch = "apple7"` and carry
// `tessera.launch_bindings` (the host buffer aliases, inputs then results).
// softmax and gelu are claimed only when an extent is dynamic -- their static
// forms belong to the shared semantic-kernel contract. Inside the claim,
// anything outside the envelope below fails closed (Decision #21a).
namespace {

struct AppleGPUDescriptorSpec {
  StringRef op;    // Graph op name
  StringRef kind;  // the descriptor family (bootstrap_prune_gap row)
  StringRef tile;  // Tile op suffix and value-call op_kind
  unsigned inputs;
  unsigned results;
};

// Must agree with `python/tessera/compiler/scheduled_apple_gpu_descriptor.py`
// (`APPLE_GPU_DESCRIPTOR_OPS`) and with the Apple GPU value lowering's symbol
// table; `tests/unit/test_apple_gpu_descriptor_migration.py` pins both.
static const AppleGPUDescriptorSpec kAppleGPUDescriptorSpecs[] = {
    {"tessera.batched_gemm", "batched_gemm", "batched_gemm", 2, 1},
    {"tessera.transpose", "transpose", "transpose", 1, 1},
    {"tessera.softmax", "dynamic_softmax", "softmax", 1, 1},
    {"tessera.gelu", "dynamic_gelu", "gelu", 1, 1},
    {"tessera.popcount", "dynamic_popcount", "popcount", 1, 1},
    {"tessera.count_nonzero", "dynamic_count_nonzero", "count_nonzero", 1, 1},
    {"tessera.top_k", "dynamic_topk", "top_k", 1, 2},
};

static const AppleGPUDescriptorSpec *appleGPUDescriptorSpec(Operation *op) {
  StringRef name = op->getName().getStringRef();
  for (const AppleGPUDescriptorSpec &spec : kAppleGPUDescriptorSpecs)
    if (spec.op == name)
      return &spec;
  return nullptr;
}

static std::string appleGPUDescriptorHash(DictionaryAttr contract) {
  std::string text;
  llvm::raw_string_ostream os(text);
  contract.print(os);
  os.flush();
  return llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(text)), true);
}

static StringRef appleGPUStorage(Type element) {
  if (element.isF32()) return "f32";
  if (element.isF16()) return "f16";
  if (element.isBF16()) return "bf16";
  if (element.isInteger(32)) return "i32";
  return "";
}

// Derive the contract for one owned Graph op, or fail with a diagnostic. The
// same function re-derives it at Schedule -> Tile, so a Schedule record that
// no longer matches its Graph parent is refused.
static FailureOr<DictionaryAttr> appleGPUDescriptorContract(Operation *op) {
  const AppleGPUDescriptorSpec *spec = appleGPUDescriptorSpec(op);
  auto fn = op->getParentOfType<func::FuncOp>();
  auto mod = op->getParentOfType<ModuleOp>();
  auto fail = [&](const Twine &message) -> FailureOr<DictionaryAttr> {
    op->emitError("Apple GPU native descriptor: " + message);
    return failure();
  };
  if (!spec || !fn || !mod)
    return fail("requires a registered owned op");
  auto target = mod->getAttrOfType<StringAttr>("tessera.target");
  auto arch = mod->getAttrOfType<StringAttr>("tessera.arch");
  if (!target || target.getValue() != "apple_gpu" || !arch ||
      arch.getValue() != "apple7")
    return fail("requires the apple7 Apple GPU target");
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

  // Every static extent and every ABI scalar is a C `int` in the runtime ABI.
  constexpr int64_t kInt32Max = std::numeric_limits<int32_t>::max();
  SmallVector<RankedTensorType> types;
  auto addType = [&](Type type) {
    auto ty = dyn_cast<RankedTensorType>(type);
    if (!ty || ty.getEncoding() || appleGPUStorage(ty.getElementType()).empty() ||
        llvm::any_of(ty.getShape(), [&](int64_t d) {
          return !ShapedType::isDynamic(d) && (d <= 0 || d > kInt32Max);
        }))
      return false;
    types.push_back(ty);
    return true;
  };
  for (Value value : op->getOperands())
    if (!addType(value.getType()))
      return fail("requires positive int32-extent f32/f16/bf16/i32 operands");
  for (auto [index, value] : llvm::enumerate(op->getResults()))
    if (!addType(value.getType()) || fn.getResultTypes()[index] != value.getType())
      return fail("requires positive int32-extent f32/f16/bf16/i32 results");

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
  auto str = [&](StringRef v) { return b.getStringAttr(v); };
  auto storageOf = [&](RankedTensorType ty) { return appleGPUStorage(ty.getElementType()); };
  auto fullyDynamic = [](RankedTensorType ty, int64_t rank) {
    return ty.getRank() == rank &&
           llvm::all_of(ty.getShape(), ShapedType::isDynamic);
  };
  auto allStatic = [&]() {
    return llvm::all_of(types, [](RankedTensorType t) { return t.hasStaticShape(); });
  };
  // An ordered ABI scalar list: [name, C type] pairs, bound by the launcher.
  SmallVector<Attribute> scalars;
  auto scalar = [&](StringRef name, StringRef type) {
    scalars.push_back(b.getArrayAttr({str(name), str(type)}));
  };
  SmallVector<NamedAttribute> semantics;
  llvm::StringSet<> allowed{"tessera.effect_kind", "schedule.artifact_hash"};
  // `axis` is optional on the Graph op; when present it must name the last
  // axis of a rank-2 operand. The contract records the normalized -1.
  auto lastAxis = [&]() -> bool {
    allowed.insert("axis");
    Attribute value = op->getAttr("axis");
    if (!value) return true;
    auto integer = dyn_cast<IntegerAttr>(value);
    return integer && (integer.getInt() == -1 || integer.getInt() == 1);
  };

  StringRef kind = spec->kind, symbol, abi, framework = "Metal";
  StringRef storage = storageOf(types[0]);
  if (kind == "batched_gemm") {
    RankedTensorType lhs = types[0], rhs = types[1], out = types[2];
    if (!allStatic() || lhs.getRank() != 3 || rhs.getRank() != 3 ||
        storage == "i32" || storageOf(rhs) != storage || storageOf(out) != storage)
      return fail("batched_gemm requires static rank-3 f32/f16/bf16 operands "
                  "and result of one storage");
    int64_t batch = lhs.getDimSize(0), m = lhs.getDimSize(1),
            k = lhs.getDimSize(2), n = rhs.getDimSize(2);
    if (rhs.getShape() != ArrayRef<int64_t>{batch, k, n} ||
        out.getShape() != ArrayRef<int64_t>{batch, m, n})
      return fail("batched_gemm requires [B,M,K] x [B,K,N] -> [B,M,N] with no "
                  "broadcasting");
    framework = "MPS";
    if (storage == "f32") {
      symbol = "tessera_apple_gpu_bmm_f32";
      abi = "tessera.apple.bmm.a_b_o_batch_m_n_k.f32.v1";
    } else if (storage == "f16") {
      symbol = "tessera_apple_gpu_bmm_f16";
      abi = "tessera.apple.bmm.a_b_o_batch_m_n_k.f16.v1";
    } else {
      symbol = "tessera_apple_gpu_bmm_bf16";
      abi = "tessera.apple.bmm.a_b_o_batch_m_n_k.bf16.v1";
    }
    semantics.push_back(b.getNamedAttr("broadcast_b", b.getBoolAttr(false)));
  } else if (kind == "transpose") {
    RankedTensorType in = types[0], out = types[1];
    int64_t rank = in.getRank();
    if (!allStatic() || (rank != 2 && rank != 3) || storage == "i32" ||
        storageOf(out) != storage || out.getRank() != rank)
      return fail("transpose requires a static rank-2/rank-3 f32/f16/bf16 "
                  "permutation of one storage");
    allowed.insert("permutation");
    SmallVector<int64_t> axes;
    if (Attribute raw = op->getAttr("permutation")) {
      auto perm = dyn_cast<DenseI64ArrayAttr>(raw);
      if (!perm) return fail("transpose permutation must be an i64 array");
      axes.assign(perm.asArrayRef().begin(), perm.asArrayRef().end());
    } else {
      for (int64_t d = rank - 1; d >= 0; --d) axes.push_back(d);
    }
    llvm::SmallDenseSet<int64_t> unique(axes.begin(), axes.end());
    if (static_cast<int64_t>(axes.size()) != rank ||
        static_cast<int64_t>(unique.size()) != rank ||
        llvm::any_of(axes, [&](int64_t a) { return a < 0 || a >= rank; }))
      return fail("transpose permutation must cover the operand rank");
    for (int64_t d = 0; d < rank; ++d)
      if (out.getDimSize(d) != in.getDimSize(axes[d]))
        return fail("transpose result shape disagrees with its permutation");
    framework = "MPSGraph";
    symbol = storage == "f32" ? "tessera_apple_gpu_mpsgraph_transpose_f32"
                              : "tessera_apple_gpu_mpsgraph_transpose_f16";
    abi = storage == "f32"   ? "tessera.apple.mpsgraph.transpose.x_o_dims_perm.f32.v1"
          : storage == "f16" ? "tessera.apple.mpsgraph.transpose.x_o_dims_perm.f16.v1"
                             : "tessera.apple.mpsgraph.transpose.x_o_dims_perm.bf16.v1";
    semantics.push_back(b.getNamedAttr("axes", b.getDenseI64ArrayAttr(axes)));
  } else if (kind == "dynamic_softmax") {
    if (!fullyDynamic(types[0], 2) || types[1] != types[0] || storage != "f32" ||
        !lastAxis())
      return fail("dynamic softmax requires ?x? f32 last-axis input and output");
    symbol = "tessera_apple_gpu_softmax_f32_status";
    abi = "tessera.apple.softmax.x_o_rows_columns.dynamic.f32.v2";
    scalar("Rows", "int32");
    scalar("Columns", "int32");
    semantics.push_back(b.getNamedAttr("axis", i64(-1)));
  } else if (kind == "dynamic_gelu") {
    if (!fullyDynamic(types[0], 2) || types[1] != types[0] || storage == "i32")
      return fail("dynamic gelu requires matching ?x? f32/f16/bf16 tensors");
    symbol = storage == "f32"   ? "tessera_apple_gpu_gelu_f32_status"
             : storage == "f16" ? "tessera_apple_gpu_gelu_f16_status"
                                : "tessera_apple_gpu_gelu_bf16_status";
    abi = storage == "f32"   ? "tessera.apple.gelu.x_o_elements.dynamic.f32.v2"
          : storage == "f16" ? "tessera.apple.gelu.x_o_elements.dynamic.f16.v2"
                             : "tessera.apple.gelu.x_o_elements.dynamic.bf16.v2";
    scalar("Elements", "int64");
    semantics.push_back(b.getNamedAttr("approximation", str("tanh")));
  } else if (kind == "dynamic_popcount") {
    if (!fullyDynamic(types[0], 1) || types[1] != types[0] || storage != "i32")
      return fail("dynamic popcount requires matching rank-1 ? i32 tensors");
    symbol = "tessera_apple_gpu_popcount_i32";
    abi = "tessera.apple.popcount.x_o_elements.dynamic.i32.v1";
    scalar("Elements", "int32");
  } else if (kind == "dynamic_count_nonzero") {
    allowed.insert("keepdims");
    auto keep = op->getAttrOfType<BoolAttr>("keepdims");
    if (!fullyDynamic(types[0], 2) || storage != "f32" ||
        !fullyDynamic(types[1], 1) || storageOf(types[1]) != "i32" ||
        !lastAxis() || (keep && keep.getValue()))
      return fail("dynamic count_nonzero requires ?x? f32 -> ? i32 over the "
                  "last axis with keepdims = false");
    symbol = "tessera_apple_gpu_count_nonzero_lastaxis_f32";
    abi = "tessera.apple.count_nonzero.x_o_outer_axis_extent.dynamic.f32_i32.v1";
    scalar("Outer", "int32");
    scalar("AxisExtent", "int32");
    semantics.append({b.getNamedAttr("axis", i64(-1)),
                      b.getNamedAttr("keepdims", b.getBoolAttr(false))});
  } else {  // dynamic_topk
    allowed.insert("k");
    auto kAttr = op->getAttrOfType<IntegerAttr>("k");
    RankedTensorType values = types[1], indices = types[2];
    int64_t k = kAttr ? kAttr.getInt() : 0;
    auto rowsByK = [&](RankedTensorType t) {
      return t.getRank() == 2 && ShapedType::isDynamic(t.getDimSize(0)) &&
             t.getDimSize(1) == k;
    };
    if (!fullyDynamic(types[0], 2) || storage != "f32" || k <= 0 ||
        k > kInt32Max || !lastAxis() || !rowsByK(values) || !rowsByK(indices) ||
        storageOf(values) != "f32" || storageOf(indices) != "i32")
      return fail("dynamic top_k requires ?x? f32 -> (?xK f32, ?xK i32) over "
                  "the last axis with a static positive K");
    symbol = "tessera_apple_gpu_topk_f32";
    abi = "tessera.apple.top_k.x_values_indices_rows_columns_k.dynamic.f32_i32.v1";
    scalar("Rows", "int32");
    scalar("Columns", "int32");
    scalar("K", "int32");
    // The kernel's ordering contract, recorded so a different policy is a
    // different contract rather than an unstated default.
    semantics.append({b.getNamedAttr("axis", i64(-1)), b.getNamedAttr("k", i64(k)),
                      b.getNamedAttr("value_order", str("descending")),
                      b.getNamedAttr("nan_policy", str("last")),
                      b.getNamedAttr("tie_policy", str("lower_index"))});
  }

  for (NamedAttribute attr : op->getAttrs()) {
    if (!allowed.contains(attr.getName().getValue()))
      return fail("has an unsupported policy attribute '" +
                  attr.getName().getValue() + "'");
    if (attr.getName() == "tessera.effect_kind" && attr.getValue() != str("pure"))
      return fail("requires a pure Graph operation");
  }

  SmallVector<Attribute> shapes, storages;
  for (RankedTensorType ty : types) {
    SmallVector<int64_t> dims;
    for (int64_t d : ty.getShape())
      dims.push_back(ShapedType::isDynamic(d) ? -1 : d);
    shapes.push_back(b.getDenseI64ArrayAttr(dims));
    storages.push_back(str(storageOf(ty)));
  }
  return b.getDictionaryAttr({
      b.getNamedAttr("kind", str(kind)),
      b.getNamedAttr("graph_op", str(spec->op)),
      b.getNamedAttr("tile_op", str(("tile." + spec->tile).str())),
      b.getNamedAttr("symbol", str(symbol)),
      b.getNamedAttr("abi", str(abi)),
      b.getNamedAttr("framework", str(framework)),
      b.getNamedAttr("inputs", i64(spec->inputs)),
      b.getNamedAttr("bindings", names),
      b.getNamedAttr("shapes", b.getArrayAttr(shapes)),
      b.getNamedAttr("storage", b.getArrayAttr(storages)),
      b.getNamedAttr("layout", str("row_major")),
      b.getNamedAttr("scalars", b.getArrayAttr(scalars)),
      b.getNamedAttr("semantics", b.getDictionaryAttr(semantics)),
  });
}

// softmax/gelu belong to this contract only in their dynamic form.
static bool appleGPUDescriptorOwns(Operation *op) {
  const AppleGPUDescriptorSpec *spec = appleGPUDescriptorSpec(op);
  if (!spec) return false;
  if (spec->kind != "dynamic_softmax" && spec->kind != "dynamic_gelu")
    return true;
  return llvm::any_of(op->getOperandTypes(), [](Type t) {
    auto ty = dyn_cast<RankedTensorType>(t);
    return !ty || !ty.hasStaticShape();
  });
}

static bool claimsAppleGPUDescriptor(ModuleOp mod) {
  auto target = mod->getAttrOfType<StringAttr>("tessera.target");
  return target && target.getValue() == "apple_gpu" &&
         mod->hasAttr("tessera.launch_bindings");
}

static LogicalResult scheduleNativeAppleGPUDescriptor(ModuleOp mod, bool &selected) {
  selected = false;
  if (!claimsAppleGPUDescriptor(mod)) return success();
  SmallVector<Operation *> ops;
  mod.walk([&](Operation *op) { if (appleGPUDescriptorOwns(op)) ops.push_back(op); });
  // A bindings-carrying module with none of these ops belongs to another
  // contract; one with several is no isolated request this ABI can serve.
  if (ops.empty()) return success();
  if (ops.size() != 1)
    return mod.emitError("Apple GPU native descriptor requires exactly one owned "
                         "operation in a module carrying launch bindings");
  Operation *op = ops.front();
  auto contract = appleGPUDescriptorContract(op);
  if (failed(contract)) return failure();
  auto fn = op->getParentOfType<func::FuncOp>();
  auto ret = dyn_cast<func::ReturnOp>(fn.getBody().front().back());
  if (fn.getBody().front().getOperations().size() != 2 || !ret ||
      ret.getOperands() != op->getResults())
    return op->emitError("Apple GPU native descriptor must return its only "
                         "operation's results in order");
  OpBuilder b(op);
  b.setInsertionPointAfter(op);
  auto hash = b.getStringAttr(appleGPUDescriptorHash(*contract));
  op->setAttr("schedule.artifact_hash", hash);
  OperationState state(op->getLoc(), "schedule.artifact");
  state.addAttribute("hash", hash);
  state.addAttribute("arch", b.getStringAttr("apple7"));
  state.addAttribute("shape_key", b.getStringAttr(
      (Twine("family=apple_gpu_descriptor;kind=") +
       cast<StringAttr>(contract->get("kind")).getValue()).str()));
  state.addAttribute("contract", *contract);
  b.create(state);
  selected = true;
  return success();
}

static LogicalResult lowerNativeAppleGPUDescriptor(ModuleOp mod, bool &selected) {
  selected = false;
  SmallVector<schedule::ArtifactOp> records;
  mod.walk([&](schedule::ArtifactOp op) {
    if (op.getShapeKey().starts_with("family=apple_gpu_descriptor;"))
      records.push_back(op);
  });
  if (records.empty()) return success();
  if (records.size() != 1)
    return mod.emitError("Apple GPU native descriptor Schedule requires one record");
  auto record = records.front();
  auto fn = record->getParentOfType<func::FuncOp>();
  if (!fn || !fn.getBody().hasOneBlock() ||
      fn.getBody().front().getOperations().size() != 3)
    return record.emitError("Apple GPU native descriptor Schedule requires its "
                            "isolated Graph parent");
  Operation *op = &fn.getBody().front().front();
  const AppleGPUDescriptorSpec *spec = appleGPUDescriptorSpec(op);
  if (!spec || !appleGPUDescriptorOwns(op))
    return record.emitError("Apple GPU native descriptor Schedule lost its Graph operation");
  auto contract = appleGPUDescriptorContract(op);
  auto ret = dyn_cast<func::ReturnOp>(fn.getBody().front().back());
  if (failed(contract) || !ret || ret.getOperands() != op->getResults() ||
      record->getAttr("contract") != *contract ||
      record.getHash() != appleGPUDescriptorHash(*contract) ||
      record.getArch() != "apple7" ||
      op->getAttr("schedule.artifact_hash") != record->getAttr("hash") ||
      record.getShapeKey() !=
          (Twine("family=apple_gpu_descriptor;kind=") + spec->kind).str())
    return record.emitError("Apple GPU native descriptor Schedule contract was altered");

  // The Tile value op the Apple GPU value lowering consumes. It carries the
  // contract itself, so the Target op is projected from the Schedule record
  // and never from a default the Graph op left implicit.
  OpBuilder b(op);
  OperationState tile(op->getLoc(), ("tile." + spec->tile).str());
  tile.addOperands(op->getOperands());
  tile.addTypes(op->getResultTypes());
  tile.addAttribute("source", b.getStringAttr(spec->op));
  tile.addAttribute("tessera.apple_gpu_descriptor_contract", *contract);
  tile.addAttribute("tessera.schedule_hash", record->getAttr("hash"));
  Operation *lowered = b.create(tile);
  op->replaceAllUsesWith(lowered->getResults());
  record.erase();
  op->erase();
  selected = true;
  return success();
}

} // namespace
