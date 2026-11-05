// E2E-REAL-6, Apple GPU domain value families (2026-11-04).
//
// One isolated static Apple GPU Graph operation -- `tessera.rl.ppo_policy_loss`,
// the four `tessera.ebm.*` value kernels (`energy_quadratic`, `langevin_step`,
// `refinement`, `partition_exact`) or the cl(3,0)
// `tessera.clifford.geometric_product` -- becomes a content-addressed Schedule
// record and then the registered Tile value op that the Apple GPU value
// lowering (`tile-to-apple_gpu{value=true}`) turns into a verified
// `tessera_apple.gpu.kernel_call`. This is the native owner of what the generic
// value branch of `apple_native.package_native` used to decide in Python for
// these six families; the retired constructor survives only as the declared
// differential oracle (`tests/_support/apple_gpu_domain_baseline.py`,
// Decision #31(a)).
//
// The consumer of every family is a hand-written MSL / MPSGraph C ABI entry in
// `apple_gpu_runtime.mm` -- a Decision #28 Tier-3 delegate reached through the
// declared Target IR op, one lowering path per Decision #31. The kernels own
// their own launch geometry, so the Schedule records the semantic and ABI
// contract only.
//
// Ownership is opt-in per module: the Graph must name
// `tessera.target = "apple_gpu"`, `tessera.arch = "apple7"` and carry
// `tessera.launch_bindings` (inputs then the result). Without the bindings
// attribute these ops pass Graph -> Schedule untouched; with it, anything
// outside the envelope below fails closed (Decision #21a): unknown attributes,
// non-static shapes, non-f32 storage, and every semantic attribute the delegate
// does not implement (Langevin `temperature`, refinement noise, a partition
// reduction other than `logsumexp`, a Clifford layout other than `blade_last`
// or a non-Euclidean signature). Every semantic value the delegate reads is
// materialized in the contract, never left to the delegate's default.
namespace {

struct AppleGPUDomainSpec {
  StringRef op;    // Graph op name
  StringRef kind;  // Tile op suffix and value-call op_kind
  unsigned minInputs;
  unsigned maxInputs;
};

// Must agree with `python/tessera/compiler/scheduled_apple_gpu_domain.py`
// (`APPLE_GPU_DOMAIN_OPS`) and with the symbols the Apple GPU value lowering
// selects; `tests/unit/test_apple_gpu_domain_migration.py` pins both.
static const AppleGPUDomainSpec kAppleGPUDomainSpecs[] = {
    {"tessera.rl.ppo_policy_loss", "ppo_policy_loss", 3, 6},
    {"tessera.ebm.energy_quadratic", "ebm_energy_quadratic", 2, 2},
    {"tessera.ebm.langevin_step", "ebm_langevin_step", 3, 3},
    {"tessera.ebm.refinement", "ebm_refinement", 2, 2},
    {"tessera.ebm.partition_exact", "ebm_partition_exact", 1, 1},
    {"tessera.clifford.geometric_product", "clifford_geometric_product", 2, 2},
};

static const AppleGPUDomainSpec *appleGPUDomainSpec(Operation *op) {
  StringRef name = op->getName().getStringRef();
  for (const AppleGPUDomainSpec &spec : kAppleGPUDomainSpecs)
    if (spec.op == name)
      return &spec;
  return nullptr;
}

static std::string appleGPUDomainHash(DictionaryAttr contract) {
  std::string text;
  llvm::raw_string_ostream os(text);
  contract.print(os);
  os.flush();
  return llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(text)), true);
}

// Derive the contract for one owned Graph op, or fail with a diagnostic. The
// same function re-derives it at Schedule -> Tile, so a Schedule record that
// no longer matches its Graph parent is refused.
static FailureOr<DictionaryAttr> appleGPUDomainContract(Operation *op) {
  const AppleGPUDomainSpec *spec = appleGPUDomainSpec(op);
  auto fn = op->getParentOfType<func::FuncOp>();
  auto mod = op->getParentOfType<ModuleOp>();
  auto fail = [&](const Twine &message) -> FailureOr<DictionaryAttr> {
    op->emitError("Apple GPU native domain kernel: " + message);
    return failure();
  };
  if (!spec || !fn || !mod)
    return fail("requires a registered owned op");
  auto target = mod->getAttrOfType<StringAttr>("tessera.target");
  auto arch = mod->getAttrOfType<StringAttr>("tessera.arch");
  if (!target || target.getValue() != "apple_gpu" || !arch ||
      arch.getValue() != "apple7")
    return fail("requires the apple7 Apple GPU target");
  unsigned inputs = op->getNumOperands();
  if (!fn.getBody().hasOneBlock() || inputs < spec->minInputs ||
      inputs > spec->maxInputs || op->getNumResults() != 1 ||
      fn.getNumArguments() != inputs || fn.getNumResults() != 1)
    return fail("requires an isolated entry whose arguments are its operands "
                "and whose result is its result");
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

  // Every element count is a C `int32_t` in the ABI. A rank-0 result is the
  // scalar reductions' single element.
  constexpr int64_t kInt32Max = std::numeric_limits<int32_t>::max();
  SmallVector<RankedTensorType> types;
  auto addType = [&](Type type) {
    auto ty = dyn_cast<RankedTensorType>(type);
    if (!ty || !ty.hasStaticShape() || ty.getEncoding() ||
        !ty.getElementType().isF32() || ty.getNumElements() <= 0 ||
        ty.getNumElements() > kInt32Max ||
        llvm::any_of(ty.getShape(), [](int64_t d) { return d <= 0; }))
      return false;
    types.push_back(ty);
    return true;
  };
  for (Value value : op->getOperands())
    if (!addType(value.getType()))
      return fail("requires static non-empty f32 operands within the int32 ABI");
  if (!addType(op->getResult(0).getType()) ||
      fn.getResultTypes()[0] != op->getResult(0).getType())
    return fail("requires a static non-empty f32 result within the int32 ABI");

  auto names = mod->getAttrOfType<ArrayAttr>("tessera.launch_bindings");
  if (!names || names.size() != types.size())
    return fail("requires one binding name per operand and the result");
  llvm::StringSet<> seen;
  for (Attribute attr : names) {
    auto name = dyn_cast<StringAttr>(attr);
    if (!name || name.getValue().empty() || !seen.insert(name.getValue()).second)
      return fail("binding names must be distinct and non-empty");
  }

  OpBuilder b(op);
  auto i64 = [&](int64_t v) { return b.getI64IntegerAttr(v); };
  RankedTensorType first = types.front(), result = types.back();
  auto same = [&](RankedTensorType ty) { return ty.getShape() == first.getShape(); };
  SmallVector<NamedAttribute> scalars;    // the ABI's integer arguments
  SmallVector<NamedAttribute> semantics;  // the materialized semantic values
  llvm::StringSet<> allowed{"tessera.effect_kind", "schedule.artifact_hash"};
  // A semantic float the delegate reads is materialized at its delegate
  // default when absent and range-checked when present (Decision #21a).
  auto real = [&](StringRef name, std::optional<double> fallback,
                  double &out) -> bool {
    allowed.insert(name);
    Attribute value = op->getAttr(name);
    auto attr = dyn_cast_or_null<FloatAttr>(value);
    if ((value && (!attr || !attr.getType().isF64())) || (!attr && !fallback))
      return false;
    out = attr ? attr.getValueAsDouble() : *fallback;
    if (!std::isfinite(out))
      return false;
    semantics.push_back(b.getNamedAttr(name, b.getF64FloatAttr(out)));
    return true;
  };
  auto text = [&](StringRef name, StringRef only) -> bool {
    allowed.insert(name);
    Attribute value = op->getAttr(name);
    auto attr = dyn_cast_or_null<StringAttr>(value);
    if (value && (!attr || attr.getValue() != only))
      return false;
    semantics.push_back(b.getNamedAttr(name, b.getStringAttr(only)));
    return true;
  };
  auto refuse = [&](StringRef name) { return op->hasAttr(name); };

  StringRef kind = spec->kind;
  StringRef symbol, abi, framework = "Metal";
  if (kind == "ppo_policy_loss") {
    // The optional tensors are positional ODS segments [new, old, adv, mask,
    // ref_logp, entropy]; the ABI takes them in that order, each present or
    // absent, so a segment is read rather than inferred from the count.
    allowed.insert("operandSegmentSizes");
    auto segments = op->getAttrOfType<DenseI32ArrayAttr>("operandSegmentSizes");
    if (!segments || segments.size() != 6 ||
        llvm::any_of(segments.asArrayRef().take_front(3),
                     [](int32_t s) { return s != 1; }) ||
        llvm::any_of(segments.asArrayRef().drop_front(3),
                     [](int32_t s) { return s != 0 && s != 1; }))
      return fail("ppo_policy_loss requires one logp_new/logp_old/advantages "
                  "and at most one of each optional side tensor");
    bool mask = segments[3] == 1, ref = segments[4] == 1,
         entropy = segments[5] == 1;
    double clip = 0.0, kl = 0.0, ent = 0.0;
    if (!llvm::all_of(ArrayRef<RankedTensorType>(types).drop_back(), same) ||
        result.getRank() != 0)
      return fail("ppo_policy_loss requires identically shaped operands and a "
                  "rank-0 result");
    if (!real("clip_epsilon", 0.2, clip) || clip <= 0.0)
      return fail("ppo_policy_loss requires a positive f64 clip_epsilon");
    if (!real("kl_coef", 0.0, kl) || kl < 0.0 || (kl != 0.0 && !ref))
      return fail("ppo_policy_loss kl_coef must be a non-negative f64 and "
                  "needs ref_logp when non-zero");
    if (!real("entropy_coef", 0.0, ent) || ent < 0.0 || (ent != 0.0 && !entropy))
      return fail("ppo_policy_loss entropy_coef must be a non-negative f64 "
                  "and needs entropy when non-zero");
    if (!text("reduction", "mean"))
      return fail("ppo_policy_loss implements reduction = \"mean\" only");
    semantics.append({b.getNamedAttr("has_mask", b.getBoolAttr(mask)),
                      b.getNamedAttr("has_ref_kl", b.getBoolAttr(ref)),
                      b.getNamedAttr("has_entropy", b.getBoolAttr(entropy))});
    bool extended = mask || ref || entropy;
    symbol = extended ? "tessera_apple_gpu_ppo_policy_loss_ex_f32"
                      : "tessera_apple_gpu_ppo_policy_loss_f32";
    abi = extended ? "mpsgraph_ppo_policy_loss_ex_f32"
                   : "mpsgraph_ppo_policy_loss_f32";
    framework = "MPSGraph";
    scalars.push_back(b.getNamedAttr("N", i64(first.getNumElements())));
  } else if (kind == "ebm_energy_quadratic") {
    if (first.getRank() != 2 || !same(types[1]) || result.getRank() != 1 ||
        result.getDimSize(0) != first.getDimSize(0))
      return fail("ebm_energy_quadratic requires x/y [B,D] and energies [B]");
    symbol = "tessera_apple_gpu_ebm_energy_quadratic_value_f32";
    abi = "msl_ebm_energy_quadratic_value_f32";
    scalars.append({b.getNamedAttr("B", i64(first.getDimSize(0))),
                    b.getNamedAttr("D", i64(first.getDimSize(1)))});
  } else if (kind == "ebm_langevin_step") {
    double eta = 0.0, scale = 0.0;
    if (!llvm::all_of(types, same))
      return fail("ebm_langevin_step requires y/grad/noise/result of one shape");
    if (!real("eta", std::nullopt, eta) || eta <= 0.0)
      return fail("ebm_langevin_step requires a positive f64 eta");
    if (!real("noise_scale", 0.0, scale) || scale < 0.0)
      return fail("ebm_langevin_step noise_scale must be a non-negative f64");
    // The delegate computes y - eta*grad + noise_scale*noise; a temperature
    // would select a different result it does not compute.
    if (refuse("temperature"))
      return fail("ebm_langevin_step does not implement temperature");
    semantics.push_back(b.getNamedAttr("has_noise", b.getBoolAttr(true)));
    symbol = "tessera_apple_gpu_ebm_langevin_step_value_f32";
    abi = "msl_ebm_langevin_step_value_f32";
    scalars.push_back(b.getNamedAttr("N", i64(first.getNumElements())));
  } else if (kind == "ebm_refinement") {
    double eta = 0.0;
    if (!llvm::all_of(types, same))
      return fail("ebm_refinement requires y/grad/result of one shape");
    if (!real("eta", std::nullopt, eta) || eta <= 0.0)
      return fail("ebm_refinement requires a positive f64 eta");
    allowed.insert("steps");
    auto steps = op->getAttrOfType<IntegerAttr>("steps");
    if (!steps || !steps.getType().isInteger(64) || steps.getInt() <= 0 ||
        steps.getInt() > kInt32Max)
      return fail("ebm_refinement requires a positive int32 i64 steps");
    // Deterministic delegate: the stochastic controls are refused, not
    // dropped.
    if (refuse("temperature") || refuse("noise_scale"))
      return fail("ebm_refinement implements the deterministic loop only");
    semantics.push_back(b.getNamedAttr("steps", i64(steps.getInt())));
    symbol = "tessera_apple_gpu_ebm_refinement_value_f32";
    abi = "msl_ebm_refinement_value_f32";
    scalars.push_back(b.getNamedAttr("N", i64(first.getNumElements())));
  } else if (kind == "ebm_partition_exact") {
    double temperature = 0.0;
    if (result.getRank() != 0)
      return fail("ebm_partition_exact requires a rank-0 result");
    if (!real("temperature", 1.0, temperature) || temperature <= 0.0)
      return fail("ebm_partition_exact requires a positive f64 temperature");
    if (!text("reduction", "logsumexp"))
      return fail("ebm_partition_exact implements reduction = \"logsumexp\" only");
    symbol = "tessera_apple_gpu_ebm_partition_exact_value_f32";
    abi = "msl_ebm_partition_exact_value_f32";
    scalars.push_back(b.getNamedAttr("N", i64(first.getNumElements())));
  } else {  // clifford_geometric_product
    allowed.insert("p");
    allowed.insert("q");
    allowed.insert("signature");
    auto p = op->getAttrOfType<IntegerAttr>("p");
    auto q = op->getAttrOfType<IntegerAttr>("q");
    if (!p || !q || p.getInt() != 3 || q.getInt() != 0)
      return fail("clifford_geometric_product implements cl(3,0) only");
    if (Attribute sig = op->getAttr("signature")) {
      auto array = dyn_cast<ArrayAttr>(sig);
      if (!array || array.size() != 3 ||
          llvm::any_of(array, [](Attribute a) {
            auto v = dyn_cast<IntegerAttr>(a);
            return !v || v.getInt() != 1;
          }))
        return fail("clifford_geometric_product implements signature [1,1,1] only");
    }
    if (!text("coefficient_layout", "blade_last"))
      return fail("clifford_geometric_product implements coefficient_layout = "
                  "\"blade_last\" only");
    if (first.getRank() < 1 || first.getDimSize(first.getRank() - 1) != 8 ||
        !llvm::all_of(types, same))
      return fail("clifford_geometric_product requires blade-last [...,8] "
                  "operands and result of one shape");
    semantics.append({b.getNamedAttr("p", i64(3)), b.getNamedAttr("q", i64(0))});
    symbol = "tessera_apple_gpu_clifford_geo_product_cl30_value_f32";
    abi = "msl_clifford_geo_product_cl30_value_f32";
    scalars.push_back(b.getNamedAttr("Batch", i64(first.getNumElements() / 8)));
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
    storages.push_back(b.getStringAttr("f32"));
  }
  return b.getDictionaryAttr({
      b.getNamedAttr("kind", b.getStringAttr(kind)),
      b.getNamedAttr("graph_op", b.getStringAttr(spec->op)),
      b.getNamedAttr("symbol", b.getStringAttr(symbol)),
      b.getNamedAttr("abi", b.getStringAttr(abi)),
      b.getNamedAttr("framework", b.getStringAttr(framework)),
      b.getNamedAttr("inputs", i64(inputs)),
      b.getNamedAttr("bindings", names),
      b.getNamedAttr("shapes", b.getArrayAttr(shapes)),
      b.getNamedAttr("storage", b.getArrayAttr(storages)),
      b.getNamedAttr("layout", b.getStringAttr("row_major")),
      b.getNamedAttr("scalars", b.getDictionaryAttr(scalars)),
      b.getNamedAttr("semantics", b.getDictionaryAttr(semantics)),
  });
}

static bool claimsAppleGPUDomain(ModuleOp mod) {
  auto target = mod->getAttrOfType<StringAttr>("tessera.target");
  return target && target.getValue() == "apple_gpu" &&
         mod->hasAttr("tessera.launch_bindings");
}

static LogicalResult scheduleNativeAppleGPUDomain(ModuleOp mod, bool &selected) {
  selected = false;
  if (!claimsAppleGPUDomain(mod)) return success();
  SmallVector<Operation *> ops;
  mod.walk([&](Operation *op) { if (appleGPUDomainSpec(op)) ops.push_back(op); });
  // A bindings-carrying module with none of these ops belongs to another
  // contract; one with several is no isolated request this ABI can serve.
  if (ops.empty()) return success();
  if (ops.size() != 1)
    return mod.emitError("Apple GPU native domain kernel requires exactly one "
                         "owned operation in a module carrying launch bindings");
  Operation *op = ops.front();
  auto contract = appleGPUDomainContract(op);
  if (failed(contract)) return failure();
  auto fn = op->getParentOfType<func::FuncOp>();
  auto ret = dyn_cast<func::ReturnOp>(fn.getBody().front().back());
  if (fn.getBody().front().getOperations().size() != 2 || !ret ||
      ret.getOperands() != op->getResults())
    return op->emitError("Apple GPU native domain kernel must return its only "
                         "operation's result");
  OpBuilder b(op);
  b.setInsertionPointAfter(op);
  auto hash = b.getStringAttr(appleGPUDomainHash(*contract));
  op->setAttr("schedule.artifact_hash", hash);
  OperationState state(op->getLoc(), "schedule.artifact");
  state.addAttribute("hash", hash);
  state.addAttribute("arch", b.getStringAttr("apple7"));
  state.addAttribute("shape_key", b.getStringAttr(
      (Twine("family=apple_gpu_domain;kind=") +
       cast<StringAttr>(contract->get("kind")).getValue()).str()));
  state.addAttribute("contract", *contract);
  b.create(state);
  selected = true;
  return success();
}

static LogicalResult lowerNativeAppleGPUDomain(ModuleOp mod, bool &selected) {
  selected = false;
  SmallVector<schedule::ArtifactOp> records;
  mod.walk([&](schedule::ArtifactOp op) {
    if (op.getShapeKey().starts_with("family=apple_gpu_domain;")) records.push_back(op);
  });
  if (records.empty()) return success();
  if (records.size() != 1)
    return mod.emitError("Apple GPU native domain kernel Schedule requires one record");
  auto record = records.front();
  auto fn = record->getParentOfType<func::FuncOp>();
  if (!fn || !fn.getBody().hasOneBlock() ||
      fn.getBody().front().getOperations().size() != 3)
    return record.emitError("Apple GPU native domain kernel Schedule requires "
                            "its isolated Graph parent");
  Operation *op = &fn.getBody().front().front();
  const AppleGPUDomainSpec *spec = appleGPUDomainSpec(op);
  if (!spec)
    return record.emitError("Apple GPU native domain kernel Schedule lost its "
                            "Graph operation");
  auto contract = appleGPUDomainContract(op);
  auto ret = dyn_cast<func::ReturnOp>(fn.getBody().front().back());
  if (failed(contract) || !ret || ret.getOperands() != op->getResults() ||
      record->getAttr("contract") != *contract ||
      record.getHash() != appleGPUDomainHash(*contract) ||
      record.getArch() != "apple7" ||
      op->getAttr("schedule.artifact_hash") != record->getAttr("hash") ||
      record.getShapeKey() !=
          (Twine("family=apple_gpu_domain;kind=") + spec->kind).str())
    return record.emitError("Apple GPU native domain kernel Schedule contract "
                            "was altered");

  // The Tile value op the Apple GPU value lowering consumes. Semantic values
  // are written explicitly from the contract, so the Target op never reads a
  // default the Graph op left implicit.
  OpBuilder b(op);
  OperationState tile(op->getLoc(), ("tile." + spec->kind).str());
  tile.addOperands(op->getOperands());
  tile.addTypes(op->getResultTypes());
  tile.addAttribute("source", b.getStringAttr(spec->op));
  for (NamedAttribute attr : cast<DictionaryAttr>(contract->get("semantics")))
    tile.addAttribute(attr.getName(), attr.getValue());
  tile.addAttribute("tessera.apple_gpu_domain_contract", *contract);
  tile.addAttribute("tessera.schedule_hash", record->getAttr("hash"));
  Operation *lowered = b.create(tile);
  op->replaceAllUsesWith(lowered->getResults());
  record.erase();
  op->erase();
  selected = true;
  return success();
}

} // namespace
