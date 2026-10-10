// Native saved-LSE tangent product. Included inside namespace tessera.
namespace {
static FailureOr<DictionaryAttr> attentionJvpContract(Operation *op) {
  const bool bias=op->getNumOperands()==10;
  auto mod=op->getParentOfType<ModuleOp>();
  auto fn=op->getParentOfType<func::FuncOp>();
  auto target=mod->getAttrOfType<StringAttr>("tessera.target");
  auto arch=mod->getAttrOfType<StringAttr>("tessera.arch");
  if (!fn || !fn.getBody().hasOneBlock() || !target || target.getValue()!="nvidia_sm120" ||
      !arch || arch.getValue()!="sm_120" || (op->getNumOperands()!=8 && !bias) || op->getNumResults()!=1)
    return op->emitError("attention JVP requires the SM120 paired tensor product"),failure();
  auto primalRef=fn->getAttrOfType<FlatSymbolRefAttr>("tessera.autodiff.forward");
  for (auto sibling:mod.getOps<func::FuncOp>()) {
    if (sibling==fn) continue;
    auto jvpRef=sibling->getAttrOfType<FlatSymbolRefAttr>("tessera.autodiff.jvp");
    if (!primalRef || primalRef.getValue()!=sibling.getSymName() ||
        !jvpRef || jvpRef.getValue()!=fn.getSymName())
      return op->emitError("attention JVP export cannot discard unrelated functions"),failure();
  }
  // Only exact source-shaped inactive zero regions may be eliminated.
  // Unrelated generated tensors or dimension reads are not export authority.
  SmallVector<Operation *> zeroOps;
  for (unsigned i=0;i<(bias?4u:3u);++i) {
    Value tangent=op->getOperand(i==3?9:i+5);
    Value primal=op->getOperand(i==3?8:i);
    if (auto generated=tangent.getDefiningOp<tensor::GenerateOp>()) {
      if (!isAttentionZeroLike(tangent,primal))
        return op->emitError("attention JVP inactive region differs from its primal extent"),failure();
      zeroOps.push_back(generated.getOperation());
      for (Value extent:generated.getDynamicExtents())
        zeroOps.push_back(extent.getDefiningOp());
    }
  }
  for (Operation &body:fn.getBody().front()) {
    if (llvm::is_contained(zeroOps,&body)) continue;
    auto name=body.getName().getStringRef();
    if (name!="tessera_attn.checkpoint_forward" && name!="tessera_attn.checkpoint_jvp" &&
        name!="arith.constant" && name!="func.return" && name!="schedule.artifact" &&
        name!="tessera.flash_attn")
      return op->emitError("attention JVP requires an isolated paired product"),failure();
  }
  if (!mod.getOps<gpu::GPUModuleOp>().empty())
    return op->emitError("attention JVP export cannot replace an existing GPU module"),failure();
  SmallVector<RankedTensorType> types;
  for (Type t:op->getOperandTypes()) {
    auto tensor=dyn_cast<RankedTensorType>(t);
    if (!tensor || tensor.getEncoding() || !tensor.getElementType().isF32())
      return op->emitError("attention JVP requires unencoded f32 tensors"),failure();
    types.push_back(tensor);
  }
  auto q=types[0],k=types[1],v=types[2];
  if (q.getRank()!=4 || k.getRank()!=4 || v.getRank()!=4)
    return op->emitError("attention JVP Q/K/V require rank four"),failure();
  SmallVector<int64_t> dims{q.getDimSize(0),q.getDimSize(1),k.getDimSize(1),
      q.getDimSize(2),k.getDimSize(2),q.getDimSize(3),v.getDimSize(3)};
  // One verified policy supplies physical capacities; symbolic sequence
  // dimensions remain symbolic in the identity and checked scalar ABI.
  auto shape = resolveNativeAttentionShape(op, dims);
  if (failed(shape)) return failure();
  auto capacities=shape->dynamicSequence?shape->bounds.asArrayRef():ArrayRef<int64_t>(dims);
  for (unsigned i=0;i<types.size();++i) {
    uint64_t bytes=4;
    for (auto entry:llvm::enumerate(types[i].getShape())) {
      int64_t d=entry.value();
      if (ShapedType::isDynamic(d)) {
        unsigned axis=entry.index();
        const bool scoreBias=bias && (i==8 || i==9);
        if (axis!=2 && !(scoreBias && axis==3))
          return op->emitError("attention JVP only sequences may be symbolic"),failure();
        d=capacities[(scoreBias && axis==3) || i==1 || i==2 || i==6 || i==7 ? 4:3];
      }
      if (d<=0 || d>65536 || bytes>uint64_t(INT64_MAX)/d)
        return op->emitError("attention JVP shape exceeds its checked byte envelope"),failure();
      bytes*=d;
    }
  }
  if (capacities[0]*capacities[1]*capacities[3]>INT32_MAX)
    return op->emitError("attention JVP row grid exceeds its launch envelope"),failure();
  auto scale=op->getAttrOfType<FloatAttr>("scale");
  auto causal=op->getAttrOfType<BoolAttr>("causal");
  if (!scale || !scale.getType().isF32() || !causal ||
      !std::isfinite(scale.getValueAsDouble()) || scale.getValueAsDouble()<=0)
    return op->emitError("attention JVP requires finite positive f32 scale"),failure();
  for (NamedAttribute attr:op->getAttrs())
    if (attr.getName()!="scale" && attr.getName()!="causal" &&
        attr.getName()!="schedule.artifact_hash")
      return op->emitError("attention JVP policy is unsupported"),failure();
  // The registered Graph verifier proves the O/LSE producer relation.
  // Check it here too: Schedule replay never substitutes an unrelated residual.
  auto forward=op->getOperand(3).getDefiningOp();
  if (!forward || forward->getName().getStringRef()!="tessera_attn.checkpoint_forward" ||
      forward->getNumOperands()!=(bias?4u:3u) || forward->getNumResults()!=2 ||
      (bias && op->getOperand(8)!=forward->getOperand(3)) ||
      op->getOperand(4)!=forward->getResult(1) ||
      forward->getAttrDictionary()!=DictionaryAttr::get(op->getContext(),{
        NamedAttribute(StringAttr::get(op->getContext(),"causal"),causal),
        NamedAttribute(StringAttr::get(op->getContext(),"scale"),scale)}))
    return op->emitError("attention JVP lost its paired forward generation"),failure();
  for (Operation &body:fn.getBody().front()) {
    if (body.getName().getStringRef()!="tessera.flash_attn") continue;
    if (body.getNumOperands()!=(bias?4u:3u) || body.getNumResults()!=1 ||
        body.getResult(0).getType()!=op->getOperand(3).getType() ||
        body.getAttrOfType<BoolAttr>("causal")!=causal)
      return op->emitError("attention JVP primal differs from its paired generation"),failure();
    for (unsigned i=0;i<(bias?4u:3u);++i)
      if (body.getOperand(i)!=op->getOperand(i==3?8:i))
        return op->emitError("attention JVP primal argument roles changed"),failure();
    auto dropout=body.getAttrOfType<FloatAttr>("dropout_p");
    auto primalScale=body.getAttrOfType<FloatAttr>("scale");
    double expectedScale=primalScale ? primalScale.getValueAsDouble() : 1.0/std::sqrt(double(dims[5]));
    if ((dropout && dropout.getValueAsDouble()!=0) ||
        float(expectedScale)!=float(scale.getValueAsDouble()))
      return op->emitError("attention JVP primal numerical policy changed"),failure();
  }
  for (unsigned i=0;i<3;++i)
    if (op->getOperand(i)!=forward->getOperand(i) || !isa<BlockArgument>(op->getOperand(i)))
      return op->emitError("attention JVP requires direct primal argument roles"),failure();
  auto returned=dyn_cast<func::ReturnOp>(fn.getBody().front().getTerminator());
  if (!returned || (returned.getNumOperands()!=1 && returned.getNumOperands()!=2) ||
      returned.getOperand(returned.getNumOperands()-1)!=op->getResult(0))
    return op->emitError("attention JVP export must return its selected tangent"),failure();
  if (returned.getNumOperands()==2 && returned.getOperand(0)!=op->getOperand(3)) {
    auto primal=returned.getOperand(0).getDefiningOp();
    if (!primal || primal->getName().getStringRef()!="tessera.flash_attn")
      return op->emitError("attention JVP export returned an unrelated primal"),failure();
  }
  SmallVector<Attribute> active;
  SmallVector<int64_t> roles;
  for (Value primal:op->getOperands().take_front(3))
    roles.push_back(cast<BlockArgument>(primal).getArgNumber());
  if (bias) {
    if (!isa<BlockArgument>(op->getOperand(8)))
      return op->emitError("attention JVP requires a direct score bias role"),failure();
    roles.push_back(cast<BlockArgument>(op->getOperand(8)).getArgNumber());
  }
  OpBuilder b(op);
  SmallVector<Value> directions(op->getOperands().slice(5,3));
  if (bias) directions.push_back(op->getOperand(9));
  for (auto entry:llvm::enumerate(directions)) {
    Value tangent=entry.value();
    if (auto arg=dyn_cast<BlockArgument>(tangent)) {
      active.push_back(b.getBoolAttr(true)); roles.push_back(arg.getArgNumber());
    }
    else {
      Value primal=op->getOperand(entry.index()==3?8:entry.index());
      if (!isAttentionZeroLike(tangent,primal))
        return op->emitError("attention JVP tangent must be a direct argument or inactive zero"),failure();
      active.push_back(b.getBoolAttr(false)); roles.push_back(-1);
    }
  }
  bool scoresActive=cast<BoolAttr>(active[0]).getValue() || cast<BoolAttr>(active[1]).getValue() ||
      (bias && cast<BoolAttr>(active[3]).getValue());
  auto contract=b.getDictionaryAttr({b.getNamedAttr("family",b.getStringAttr("attention_checkpoint_jvp")),
      b.getNamedAttr("shape",b.getDenseI64ArrayAttr(dims)),b.getNamedAttr("scale",scale),
      b.getNamedAttr("causal",causal),b.getNamedAttr("active",b.getArrayAttr(active)),
      b.getNamedAttr("argument_roles",b.getDenseI64ArrayAttr(roles)),
      b.getNamedAttr("target",target),b.getNamedAttr("arch",arch),
      b.getNamedAttr("algorithm",b.getStringAttr(scoresActive ? "cooperative_saved_lse_moments_v1" : "cooperative_saved_lse_value_linear_v1")),
      b.getNamedAttr("workgroup_size",b.getI64IntegerAttr(128)),
      b.getNamedAttr("ownership",b.getStringAttr("private_saved_generation_distinct_tangent"))});
  if (shape->dynamicSequence) {
    NamedAttrList attrs(contract);
    attrs.set("shape_bounds",shape->bounds);
    attrs.set("shape_policy",b.getStringAttr("bounded_sequences_v1"));
    contract=attrs.getDictionary(op->getContext());
  }
  if (bias) {
    NamedAttrList attrs(contract);
    attrs.set("bias_shape",b.getDenseI64ArrayAttr(types[8].getShape()));
    contract=attrs.getDictionary(op->getContext());
  }
  return contract;
}
static std::string attentionJvpHash(DictionaryAttr c) {
  std::string text;llvm::raw_string_ostream os(text);c.print(os);os.flush();
  return llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(text)),true);
}
static LogicalResult scheduleNativeAttentionJvp(ModuleOp mod,bool &selected) {
  SmallVector<Operation*> ops;
  mod.walk([&](Operation *op){if(op->getName().getStringRef()=="tessera_attn.checkpoint_jvp")ops.push_back(op);});
  selected=!ops.empty();
  if (!selected) return success();
  if (ops.size()!=1) return mod.emitError("attention JVP export requires exactly one product");
  auto c=attentionJvpContract(ops[0]);if(failed(c))return failure();
  OpBuilder b(ops[0]);b.setInsertionPointAfter(ops[0]);
  auto hash=b.getStringAttr(attentionJvpHash(*c));
  ops[0]->setAttr("schedule.artifact_hash",hash);
  OperationState record(ops[0]->getLoc(),"schedule.artifact");
  record.addAttribute("hash",hash);record.addAttribute("arch",b.getStringAttr("sm_120"));
  record.addAttribute("shape_key",b.getStringAttr("family=attention_checkpoint_jvp"));
  record.addAttribute("contract",*c);b.create(record);
  return success();
}
static LogicalResult lowerNativeAttentionJvp(ModuleOp mod,bool &selected) {
  SmallVector<schedule::ArtifactOp> records;
  mod.walk([&](schedule::ArtifactOp op){if(op.getShapeKey()=="family=attention_checkpoint_jvp")records.push_back(op);});
  selected=!records.empty();if(!selected)return success();
  if(records.size()!=1)return mod.emitError("attention JVP Schedule requires exactly one product");
  auto record=records[0];auto fn=record->getParentOfType<func::FuncOp>();
  if (!fn || !fn.getBody().hasOneBlock())
    return record.emitError("attention JVP Schedule requires a single-block function");
  Operation *graph=nullptr;
  fn.walk([&](Operation *op){if(op->getName().getStringRef()=="tessera_attn.checkpoint_jvp")graph=op;});
  if(!graph)return record.emitError("attention JVP Schedule lost its producer");
  auto c=attentionJvpContract(graph);if(failed(c))return failure();
  if(record->getAttrs().size()!=4 || record.getArch()!="sm_120" || record->getAttr("contract")!=*c ||
      record.getHash()!=attentionJvpHash(*c) ||
      graph->getAttr("schedule.artifact_hash")!=record->getAttr("hash"))
    return record.emitError("attention JVP Schedule contract changed after hashing");
  auto dims=cast<DenseI64ArrayAttr>((*c).get("shape")).asArrayRef();
  auto bounds=dyn_cast_or_null<DenseI64ArrayAttr>((*c).get("shape_bounds"));
  const bool dynamic=bool(bounds);
  auto capacities=dynamic?bounds.asArrayRef():dims;
  auto active=cast<ArrayAttr>((*c).get("active"));
  const bool bias=graph->getNumOperands()==10;
  bool scoresActive=cast<BoolAttr>(active[0]).getValue() || cast<BoolAttr>(active[1]).getValue() ||
      (bias && cast<BoolAttr>(active[3]).getValue());
  auto biasShape=bias?cast<DenseI64ArrayAttr>((*c).get("bias_shape")).asArrayRef():ArrayRef<int64_t>();
  auto scale=cast<FloatAttr>((*c).get("scale"));
  bool causal=cast<BoolAttr>((*c).get("causal")).getValue();
  SmallVector<int64_t> shape(dims.begin(),dims.end());
  SmallVector<RankedTensorType> tensorTypes;
  for(Type t:graph->getOperandTypes())tensorTypes.push_back(cast<RankedTensorType>(t));
  tensorTypes.push_back(cast<RankedTensorType>(graph->getResult(0).getType()));
  SmallVector<StringRef> names{"q","k","v","primal","lse","dq","dk","dv"};
  if (bias) {names.push_back("bias");names.push_back("dbias");}
  names.push_back("tangent");
  llvm::json::Array specs;
  for(unsigned i=0;i<tensorTypes.size();++i) {
    llvm::json::Array extents;
    for (auto entry:llvm::enumerate(tensorTypes[i].getShape())) {
      if (!ShapedType::isDynamic(entry.value())) extents.push_back(entry.value());
      else {
        const bool keyAxis=(i==1 || i==2 || i==6 || i==7) ||
                           ((names[i]=="bias" || names[i]=="dbias") && entry.index()==3);
        extents.push_back(keyAxis?"key_size":"query_size");
      }
    }
    specs.push_back(llvm::json::Object{{"kind","tensor"},{"name",names[i]},{"dtype","fp32"},
        {"shape",std::move(extents)},{"writable",i==tensorTypes.size()-1}});
  }
  specs.push_back(llvm::json::Object{{"kind","index"},{"name","scratch"},{"minimum",128},{"maximum",128}});
  if (dynamic) {
    StringRef scalarNames[]{"query_size","key_size"};
    for (auto entry:llvm::enumerate(scalarNames)) {
      unsigned axis=entry.index()+3;
      specs.push_back(llvm::json::Object{{"kind","index"},{"name",entry.value()},
        {"minimum",ShapedType::isDynamic(dims[axis])?1:dims[axis]},
        {"maximum",capacities[axis]}});
    }
  }
  llvm::json::Array grid;
  if (dynamic) grid.push_back(llvm::json::Object{{"product",llvm::json::Array{dims[0],dims[1],"query_size"}}});
  else grid.push_back(dims[0]*dims[1]*dims[3]);
  grid.push_back(1);grid.push_back(1);
  llvm::json::Object manifest{{"schema",dynamic?2:1},{"arguments",std::move(specs)},
      {"grid",std::move(grid)},{"block",llvm::json::Array{128,1,1}}};
  std::string json;llvm::raw_string_ostream jos(json);jos<<llvm::json::Value(std::move(manifest));jos.flush();
  OpBuilder b(mod.getContext());auto loc=graph->getLoc();
  mod->setAttr("tessera.native_tensor_contract",b.getStringAttr(json));
  mod->setAttr("tessera.attention_jvp_schedule_hash",record->getAttr("hash"));
  mod->setAttr("tessera.attention_jvp_contract",*c);
  uint32_t bits=scale.getValue().bitcastToAPInt().getZExtValue();
  uint8_t bytes[]{uint8_t(bits>>24),uint8_t(bits>>16),uint8_t(bits>>8),uint8_t(bits)};
  std::string identity;llvm::raw_string_ostream ios(identity);
  ios<<"{";
  bool broadcast=false;
  if (bias) {
    ios<<"\"bias\":\"exact_f32[B,Hq,Sq,Sk]\",";
    const int64_t scores[4]={dims[0],dims[1],dims[3],dims[4]};
    for(unsigned i=0;i<4;++i)broadcast|=biasShape[i]!=scores[i];
    if(broadcast) {
      ios<<"\"bias_gradient_reduction\":\"physical_owner_lexicographic_bhqk_v1\",\"bias_shape\":[";
      for(unsigned i=0;i<4;++i){if(i)ios<<",";ios<<biasShape[i];}
      ios<<"],";
    }
  }
  ios<<"\"causal\":"<<(causal?"true":"false")
     <<",\"lse\":\"natural_log\",\"mask_alignment\":\"end_aligned_v1\",\"scale_f32_bits\":\""
     <<llvm::toHex(ArrayRef<uint8_t>(bytes),true)
     <<"\",\"schema\":\""+std::string(dynamic?"tessera.attention_checkpoint.bounded_sequences.v1":broadcast?"tessera.attention_checkpoint.broadcast.v1":"tessera.attention_checkpoint.v1")+"\",\"shape\":[";
  for(unsigned i=0;i<7;++i){if(i)ios<<",";ios<<dims[i];}
  ios<<"]";
  if (dynamic) {
    ios<<",\"shape_bounds\":[";
    for(unsigned i=0;i<7;++i){if(i)ios<<",";ios<<capacities[i];}
    ios<<"]";
  }
  ios<<",\"storage\":\"f32\"}";ios.flush();
  mod->setAttr("tessera.attention_checkpoint_identity",b.getStringAttr(
      llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(identity)),true)));
  b.setInsertionPointToEnd(mod.getBody());
  auto gm=gpu::GPUModuleOp::create(b,loc,"attention_jvp");
  b.setInsertionPointToStart(&gm.getBodyRegion().front());
  auto ptr=LLVM::LLVMPointerType::get(mod.getContext(),1);
  SmallVector<Type> abi(tensorTypes.size(),ptr);abi.push_back(b.getIndexType());
  if (dynamic) {abi.push_back(b.getIndexType());abi.push_back(b.getIndexType());}
  auto kernel=gpu::GPUFuncOp::create(b,loc,"saved_lse_jvp",b.getFunctionType(abi,{}));
  kernel.setKernelAttr(b.getUnitAttr());
  kernel->setAttr("tessera.schedule_hash",record->getAttr("hash"));
  b.setInsertionPointToStart(&kernel.getBody().front());
  auto args=kernel.getArguments();
  auto ci=[&](int64_t n)->Value{return arith::ConstantIndexOp::create(b,loc,n);};
  auto c64=[&](int64_t n)->Value{return arith::ConstantIntOp::create(b,loc,n,64);};
  auto add=[&](Value x,Value y)->Value{return arith::AddIOp::create(b,loc,x,y);};
  auto mul=[&](Value x,Value y)->Value{return arith::MulIOp::create(b,loc,x,y);};
  auto div=[&](Value x,Value y)->Value{return arith::DivUIOp::create(b,loc,x,y);};
  auto rem=[&](Value x,Value y)->Value{return arith::RemUIOp::create(b,loc,x,y);};
  auto cast64=[&](Value x)->Value{return arith::IndexCastOp::create(b,loc,b.getI64Type(),x);};
  auto fadd=[&](Value x,Value y)->Value{return arith::AddFOp::create(b,loc,x,y);};
  auto fmul=[&](Value x,Value y)->Value{return arith::MulFOp::create(b,loc,x,y);};
  auto fsub=[&](Value x,Value y)->Value{return arith::SubFOp::create(b,loc,x,y);};
  auto load=[&](unsigned arg,Value offset)->Value{
    Value p=LLVM::GEPOp::create(b,loc,ptr,b.getF32Type(),args[arg],ValueRange{offset});
    return LLVM::LoadOp::create(b,loc,b.getF32Type(),p,4);
  };
  Value tid=gpu::ThreadIdOp::create(b,loc,gpu::Dimension::x);
  Value row=cast64(gpu::BlockIdOp::create(b,loc,gpu::Dimension::x));
  Value t=cast64(tid),z=ci(0),one=ci(1),width=ci(128);
  Value zero=arith::ConstantFloatOp::create(b,loc,b.getF32Type(),APFloat(0.0f));
  Value sf=arith::ConstantOp::create(b,loc,scale);
  Value log2e=arith::ConstantFloatOp::create(b,loc,b.getF32Type(),APFloat(1.4426950408889634f));
  Value sq=dynamic?cast64(args[tensorTypes.size()+1]):c64(dims[3]);
  Value sk=dynamic?cast64(args[tensorTypes.size()+2]):c64(dims[4]);
  Value keyExtent=dynamic?args[tensorTypes.size()+2]:ci(dims[4]);
  Value qi=rem(row,sq),bh=div(row,sq);
  Value head=rem(bh,c64(dims[1])),batch=div(bh,c64(dims[1]));
  Value kvhead=div(head,c64(dims[1]/dims[2]));
  Value kvbase=mul(add(mul(batch,c64(dims[2])),kvhead),sk);
  Value qbase=mul(row,c64(dims[5])),obase=mul(row,c64(dims[6]));
  Value gap;
  if (dynamic) {
    Value positive=arith::CmpIOp::create(b,loc,arith::CmpIPredicate::ugt,sk,sq);
    gap=arith::SelectOp::create(b,loc,positive,arith::SubIOp::create(b,loc,sk,sq),c64(0));
  } else gap=c64(std::max<int64_t>(dims[4]-dims[3],0));
  Value limit=add(qi,gap);
  Value L=load(4,row);
  auto scratchType=MemRefType::get({ShapedType::kDynamic},b.getF32Type());
  SmallVector<Value> scratch;
  for(unsigned i=0;i<(scoresActive?2u:1u);++i) {
    Value buffer=memref::AllocaOp::create(b,loc,scratchType,ValueRange{args[tensorTypes.size()]});
    OperationState shared(loc,"tile.alloc_shared");shared.addOperands(buffer);b.create(shared);
    scratch.push_back(buffer);
  }
  auto cols=scf::ForOp::create(b,loc,z,ci(dims[6]),one);
  b.setInsertionPointToStart(cols.getBody());
  Value col=cast64(cols.getInductionVar());
  auto keys=scf::ForOp::create(b,loc,tid,keyExtent,width,ValueRange{zero,zero});
  b.setInsertionPointToStart(keys.getBody());
  Value key=cast64(keys.getInductionVar());
  Value legal=causal?Value(arith::CmpIOp::create(b,loc,arith::CmpIPredicate::ule,key,limit))
      :Value(arith::ConstantOp::create(b,loc,b.getBoolAttr(true)));
  auto mask=scf::IfOp::create(b,loc,TypeRange{b.getF32Type(),b.getF32Type()},legal,true);
  b.setInsertionPointToStart(mask.thenBlock());
  Value kr=add(kvbase,key),kbase=mul(kr,c64(dims[5]));
  auto dot=scf::ForOp::create(b,loc,z,ci(dims[5]),one,ValueRange{zero,zero});
  b.setInsertionPointToStart(dot.getBody());
  Value axis=cast64(dot.getInductionVar());
  Value qv=load(0,add(qbase,axis)),kv=load(1,add(kbase,axis));
  Value dq=cast<BoolAttr>(active[0]).getValue()?load(5,add(qbase,axis)):zero;
  Value dk=cast<BoolAttr>(active[1]).getValue()?load(6,add(kbase,axis)):zero;
  scf::YieldOp::create(b,loc,ValueRange{fadd(dot.getRegionIterArgs()[0],fmul(qv,kv)),
      fadd(dot.getRegionIterArgs()[1],fadd(fmul(dq,kv),fmul(qv,dk)))});
  b.setInsertionPointAfter(dot);
  Value score=fmul(dot.getResult(0),sf),direction=fmul(dot.getResult(1),sf);
  if (bias) {
    Value coordinates[4]={batch,head,qi,key},offset=c64(0);
    for(unsigned i=0;i<4;++i) {
      Value pitch=ShapedType::isDynamic(biasShape[i])?(i==2?sq:sk):c64(biasShape[i]);
      offset=add(mul(offset,pitch),biasShape[i]==1?c64(0):coordinates[i]);
    }
    score=fadd(score,load(8,offset));
    if(cast<BoolAttr>(active[3]).getValue())direction=fadd(direction,load(9,offset));
  }
  Value probability=math::Exp2Op::create(b,loc,fmul(fsub(score,L),log2e));
  Value vi=add(mul(kr,c64(dims[6])),col);
  Value vv=scoresActive?load(2,vi):zero,dv=cast<BoolAttr>(active[2]).getValue()?load(7,vi):zero;
  Value pd=fmul(probability,direction);
  scf::YieldOp::create(b,loc,ValueRange{fadd(keys.getRegionIterArgs()[0],pd),
      fadd(keys.getRegionIterArgs()[1],fadd(fmul(pd,vv),fmul(probability,dv)))});
  b.setInsertionPointToStart(mask.elseBlock());
  scf::YieldOp::create(b,loc,keys.getRegionIterArgs());
  b.setInsertionPointAfter(mask);scf::YieldOp::create(b,loc,mask.getResults());
  b.setInsertionPointAfter(keys);
  for(unsigned i=0;i<scratch.size();++i)memref::StoreOp::create(b,loc,keys.getResult(scoresActive?i:1),scratch[i],ValueRange{tid});
  gpu::BarrierOp::create(b,loc);
  for(int64_t stride=64;stride>=1;stride>>=1) {
    Value s=ci(stride);
    auto combine=scf::IfOp::create(b,loc,arith::CmpIOp::create(b,loc,arith::CmpIPredicate::ult,tid,s),false);
    b.setInsertionPointToStart(combine.thenBlock());
    for(Value buffer:scratch) {
      Value x=memref::LoadOp::create(b,loc,buffer,ValueRange{tid});
      Value y=memref::LoadOp::create(b,loc,buffer,ValueRange{add(tid,s)});
      memref::StoreOp::create(b,loc,fadd(x,y),buffer,ValueRange{tid});
    }
    b.setInsertionPointAfter(combine);gpu::BarrierOp::create(b,loc);
  }
  auto leader=scf::IfOp::create(b,loc,arith::CmpIOp::create(b,loc,arith::CmpIPredicate::eq,t,c64(0)),false);
  b.setInsertionPointToStart(leader.thenBlock());
  Value product=memref::LoadOp::create(b,loc,scratch[scoresActive?1:0],ValueRange{z});
  Value oi=add(obase,col);
  Value result=product;
  if (scoresActive) {
    Value moment=memref::LoadOp::create(b,loc,scratch[0],ValueRange{z});
    result=fsub(product,fmul(load(3,oi),moment));
  }
  Value out=LLVM::GEPOp::create(b,loc,ptr,b.getF32Type(),args[tensorTypes.size()-1],ValueRange{oi});
  LLVM::StoreOp::create(b,loc,result,out,4);
  b.setInsertionPointAfter(leader);gpu::BarrierOp::create(b,loc);
  b.setInsertionPointAfter(cols);gpu::ReturnOp::create(b,loc,ValueRange{});
  SmallVector<func::FuncOp> old;
  for(auto f:mod.getOps<func::FuncOp>())old.push_back(f);
  for(auto f:old)f.erase();
  return success();
}
} // namespace
