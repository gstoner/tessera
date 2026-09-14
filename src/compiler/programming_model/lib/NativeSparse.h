// Native logical matmul -> checked sparse Schedule kernel. AD runs before this
// physical specialization; no derivative is taken through index selection.
namespace {
static FailureOr<std::string> sparseGraphKernel(ModuleOp mod) {
  auto policy = mod->getAttrOfType<StringAttr>("tessera.sparse_policy");
  if (!policy || (policy.getValue() != "checked_2to4" && policy.getValue() != "auto_2to4"))
    return mod.emitError("sparse lowering requires checked_2to4 policy"), failure();
  bool automatic = policy.getValue() == "auto_2to4";
  auto target = mod->getAttrOfType<StringAttr>("tessera.target");
  auto arch = mod->getAttrOfType<StringAttr>("tessera.arch");
  if (!target || target.getValue() != "rocm" || !arch || arch.getValue() != "gfx1201")
    return mod.emitError("sparse lowering requires gfx1201"), failure();
  for (auto attr : mod->getAttrs()) {
    auto name=attr.getName().getValue();
    if (name!="tessera.sparse_policy" && name!="tessera.target" && name!="tessera.arch" &&
        name!="tessera.ir.version" && name!="tessera.frontend.authority" &&
        name!="tessera.autodiff" && name!="tessera.autodiff.wrt" && name!="tessera.autodiff.wrt_indices")
      return mod.emitError("sparse lowering cannot discard module policy"), failure();
  }
  SmallVector<func::FuncOp> functions(mod.getOps<func::FuncOp>());
  if (functions.size() != 1 || mod.getBody()->getOperations().size()!=1 || !functions[0].getBody().hasOneBlock())
    return mod.emitError("sparse lowering requires one single-block entry"), failure();
  auto fn = functions[0];
  if (fn.getNumArguments() != 2 || fn.getNumResults() != 1 || fn.front().getOperations().size() != 2)
    return fn.emitError("sparse lowering requires isolated matmul and return"), failure();
  for (auto attr : fn->getAttrs()) {
    auto name=attr.getName().getValue();
    if (name=="tessera.autodiff" || name=="tessera.autodiff.wrt" || name=="tessera.autodiff.wrt_indices") {
      if (mod->getAttr(name)==attr.getValue()) continue;
      return fn.emitError("sparse logical AD declarations disagree"), failure();
    }
    if (name!="sym_name" && name!="function_type" && name!="arg_attrs" &&
        name!="tessera.frontend.authority" && name!="tessera.structured_cfg.schema" &&
        name!="tessera.structured_cfg.digest" && name!="tessera.structured_cfg.blocks")
      return fn.emitError("sparse lowering cannot discard function policy"), failure();
  }
  Operation &op = fn.front().front();
  auto ret = dyn_cast<func::ReturnOp>(fn.front().back());
  if ((op.getName().getStringRef() != "tessera.matmul" && op.getName().getStringRef() != "tessera.gemm") ||
      op.getNumOperands() != 2 || op.getNumResults() != 1 || !ret ||
      op.getOperand(0) != fn.getArgument(0) || op.getOperand(1) != fn.getArgument(1) ||
      ret.getNumOperands() != 1 || ret.getOperand(0) != op.getResult(0))
    return fn.emitError("sparse lowering requires direct ordered matmul"), failure();
  for (StringRef name : {"transposeA", "transposeB"})
    if (auto flag=op.getAttrOfType<BoolAttr>(name); flag && flag.getValue())
      return op.emitError("sparse lowering requires untransposed logical matrices"), failure();
  for (auto attr : op.getAttrs()) {
    auto name = attr.getName().getValue();
    auto value = dyn_cast<StringAttr>(attr.getValue());
    if (name=="transposeA" || name=="transposeB") {
      auto flag=dyn_cast<BoolAttr>(attr.getValue());
      if (flag && !flag.getValue()) continue;
    }
    if (name=="tessera.effect_kind" && value && value.getValue()=="pure") continue;
    if ((name != "activation" && name != "epilogue") || !value || value.getValue() != "none")
      return op.emitError("sparse lowering cannot discard operation policies"), failure();
  }
  auto a = dyn_cast<RankedTensorType>(fn.getArgument(0).getType());
  auto b = dyn_cast<RankedTensorType>(fn.getArgument(1).getType());
  auto c = dyn_cast<RankedTensorType>(op.getResult(0).getType());
  if (!a || !b || !c || a.getRank()!=2 || b.getRank()!=2 || c.getRank()!=2 ||
      !a.hasStaticShape() || !b.hasStaticShape() || !c.hasStaticShape() ||
      a.getEncoding() || b.getEncoding() || c.getEncoding() ||
      !(a.getElementType().isF16() || a.getElementType().isBF16()) || b.getElementType()!=a.getElementType() ||
      !(c.getElementType().isF32() || c.getElementType()==a.getElementType()) || fn.getResultTypes()[0]!=c)
    return fn.emitError("sparse lowering requires plain static half tensors"), failure();
  int64_t m=a.getDimSize(0), k=a.getDimSize(1), n=b.getDimSize(1);
  if (m<=0 || n<=0 || k<=0 || m>256 || n>256 || k>256 || m%16 || n%16 || k%32 ||
      b.getDimSize(0)!=k || c.getDimSize(0)!=m || c.getDimSize(1)!=n)
    return fn.emitError("sparse lowering requires compatible tiled extents <=256"), failure();
  for (unsigned i=0;i<2;++i) if (auto attrs=fn.getArgAttrDict(i)) for (auto attr : attrs) {
    auto name=attr.getName().getValue();
    if (name=="tessera.layout") {
      auto layout=dyn_cast<StringAttr>(attr.getValue());
      if (layout && layout.getValue()=="row_major") continue;
    }
    if (name=="tessera.dim_names") {
      auto names=dyn_cast<ArrayAttr>(attr.getValue());
      if (names && names.size()==2 && llvm::all_of(names,[](Attribute x){return isa<StringAttr>(x);})) continue;
    }
    return fn.emitError("sparse lowering cannot discard argument policy"), failure();
  }
  std::string e=a.getElementType().isF16()?"f16":"bf16";
  std::string out=c.getElementType().isF32()?"f32":e;
  std::string text;
  llvm::raw_string_ostream s(text);
  auto v=[&](StringRef name,StringRef value){s<<"%"<<name<<" = "<<value<<"\n";};
  s<<"module attributes {gpu.container_module} { gpu.module @sparse {\n"
   <<"gpu.func @probe(%a: memref<"<<m*k<<"x"<<e<<">, %b: memref<"<<k*n<<"x"<<e
   <<">, %out: memref<"<<m*n<<"x"<<out<<">, %status: memref<"<<(m/16)*(n/16)*32
   <<"xi32>) kernel attributes {gpu.known_block_size = array<i32: 32, 1, 1>} {\n";
  for(int i=0;i<=32;++i) s<<"%c"<<i<<" = arith.constant "<<i<<" : index\n";
  s<<"%kn = arith.constant "<<k<<" : index\n%nn = arith.constant "<<n<<" : index\n%nt = arith.constant "<<n/16<<" : index\n";
  v("zbits","arith.constant 0 : i16"); v("z32","arith.constant 0 : i32"); v("one32","arith.constant 1 : i32");
  v("yes","arith.constant true"); v("no","arith.constant false"); v("zc","arith.constant dense<0.0> : vector<8xf32>");
  v("lane","gpu.thread_id x");v("tile","gpu.block_id x");
  v("tm","arith.divui %tile, %nt : index");v("tn","arith.remui %tile, %nt : index");
  v("row0","arith.muli %tm, %c16 : index");v("col0","arith.muli %tn, %c16 : index");
  v("low","arith.remui %lane, %c16 : index");v("half","arith.divui %lane, %c16 : index");
  v("arow","arith.addi %row0, %low : index");v("abase","arith.muli %arow, %kn : index");
  v("bcol","arith.addi %col0, %low : index");v("half8","arith.muli %half, %c8 : index");
  s<<"%loop:2 = scf.for %kk = %c0 to %kn step %c32 iter_args(%acc = %zc, %valid = %yes) -> (vector<8xf32>, i1) {\n";
  std::string idx="%z32", valid="%valid";
  SmallVector<std::string> av;
  for(int r=0;r<4;++r) {
    auto g="g"+std::to_string(r);
    s<<"%"<<g<<"a = arith.addi %kk, %half8 : index\n%"<<g<<"b = arith.addi %"<<g<<"a, %c"<<(r/2)*16+(r%2)*4<<" : index\n%"<<g<<" = arith.addi %abase, %"<<g<<"b : index\n";
    for(int j=0;j<4;++j) {auto x=g+"v"+std::to_string(j);
      s<<"%"<<x<<"i = arith.addi %"<<g<<", %c"<<j<<" : index\n%"<<x<<" = memref.load %a[%"<<x<<"i] : memref<"<<m*k<<"x"<<e<<">\n%"<<x<<"bits = arith.bitcast %"<<x<<" : "<<e<<" to i16\n%"<<x<<"z = arith.cmpi eq, %"<<x<<"bits, %zbits : i16\n";
    }
    std::string first="%"+g+"v0", second="%"+g+"v1", code="%z32", found="%no";
    int pair=0;
    for(int i=0;i<4;++i) for(int j=i+1;j<4;++j) {
      int other[2],count=0;for(int q=0;q<4;++q)if(q!=i&&q!=j)other[count++]=q;
      auto p="p"+std::to_string(r)+"_"+std::to_string(pair++);
      s<<"%"<<p<<"ok = arith.andi %"<<g<<"v"<<other[0]<<"z, %"<<g<<"v"<<other[1]<<"z : i1\n%"<<p<<"code = arith.constant "<<((i|(j<<2))<<(4*r))<<" : i32\n"
       <<"%"<<p<<"x = arith.select %"<<p<<"ok, %"<<g<<"v"<<i<<", "<<first<<" : "<<e<<"\n%"<<p<<"y = arith.select %"<<p<<"ok, %"<<g<<"v"<<j<<", "<<second<<" : "<<e<<"\n"
       <<"%"<<p<<"idx = arith.select %"<<p<<"ok, %"<<p<<"code, "<<code<<" : i32\n%"<<p<<"found = arith.ori %"<<p<<"ok, "<<found<<" : i1\n";
      first="%"+p+"x";second="%"+p+"y";code="%"+p+"idx";found="%"+p+"found";
    }
    av.push_back(first);av.push_back(second);
    s<<"%idx"<<r<<" = arith.ori "<<idx<<", "<<code<<" : i32\n%valid"<<r<<" = arith.andi "<<valid<<", "<<found<<" : i1\n";
    idx="%idx"+std::to_string(r);valid="%valid"+std::to_string(r);
  }
  s<<"%av = vector.from_elements ";for(unsigned i=0;i<av.size();++i)s<<(i?", ":"")<<av[i];s<<" : vector<8x"<<e<<">\n";
  for(int j=0;j<16;++j) s<<"%br"<<j<<"a = arith.addi %kk, %half8 : index\n%br"<<j<<" = arith.addi %br"<<j<<"a, %c"<<(j/8)*16+j%8<<" : index\n%bi"<<j<<"a = arith.muli %br"<<j<<", %nn : index\n%bi"<<j<<" = arith.addi %bi"<<j<<"a, %bcol : index\n%b"<<j<<" = memref.load %b[%bi"<<j<<"] : memref<"<<k*n<<"x"<<e<<">\n";
  s<<"%bv = vector.from_elements ";for(int j=0;j<16;++j)s<<(j?", ":"")<<"%b"<<j;s<<" : vector<16x"<<e<<">\n";
  if (automatic) {
    // Every lane participates. The branch is wave-uniform, as required by MMA.
    s<<"%flag0 = arith.extui "<<valid<<" : i1 to i32\n%wave_width = arith.constant 32 : i32\n";
    int previous=0;
    for (int offset : {1,2,4,8,16}) {
      s<<"%off"<<offset<<" = arith.constant "<<offset<<" : i32\n"
       <<"%shuffle"<<offset<<", %shuffle_ok"<<offset<<" = gpu.shuffle xor %flag"<<previous<<", %off"<<offset<<", %wave_width : i32\n"
       <<"%flag"<<offset<<" = arith.andi %flag"<<previous<<", %shuffle"<<offset<<" : i32\n";
      previous=offset;
    }
    s<<"%all_sparse = arith.cmpi ne, %flag16, %z32 : i32\n%selected = scf.if %all_sparse -> (vector<8xf32>) {\n";
  }
  s<<"%result = schedule.sparse_mma %av, %bv, %acc, "<<idx<<" {arch = \"gfx1201\"} : vector<8x"<<e<<">, vector<16x"<<e<<">, vector<8xf32> -> vector<8xf32>\n";
  if (automatic) {
    s<<"scf.yield %result : vector<8xf32>\n} else {\n";
    for(int j=0;j<8;++j) {
      s<<"%dr"<<j<<"a = arith.addi %row0, %half8 : index\n%dr"<<j<<" = arith.addi %dr"<<j<<"a, %c"<<j<<" : index\n"
       <<"%dbase"<<j<<" = arith.muli %dr"<<j<<", %kn : index\n"
       <<"%seed"<<j<<" = vector.extract %acc["<<j<<"] : f32 from vector<8xf32>\n"
       <<"%dense"<<j<<" = scf.for %dk"<<j<<" = %c0 to %c32 step %c1 iter_args(%sum"<<j<<" = %seed"<<j<<") -> (f32) {\n"
       <<"%colk"<<j<<" = arith.addi %kk, %dk"<<j<<" : index\n"
       <<"%dai"<<j<<" = arith.addi %dbase"<<j<<", %colk"<<j<<" : index\n"
       <<"%dbi"<<j<<"a = arith.muli %colk"<<j<<", %nn : index\n%dbi"<<j<<" = arith.addi %dbi"<<j<<"a, %bcol : index\n"
       <<"%da"<<j<<" = memref.load %a[%dai"<<j<<"] : memref<"<<m*k<<"x"<<e<<">\n"
       <<"%db"<<j<<" = memref.load %b[%dbi"<<j<<"] : memref<"<<k*n<<"x"<<e<<">\n"
       <<"%daf"<<j<<" = arith.extf %da"<<j<<" : "<<e<<" to f32\n%dbf"<<j<<" = arith.extf %db"<<j<<" : "<<e<<" to f32\n"
       <<"%product"<<j<<" = arith.mulf %daf"<<j<<", %dbf"<<j<<" : f32\n%sum_next"<<j<<" = arith.addf %sum"<<j<<", %product"<<j<<" : f32\nscf.yield %sum_next"<<j<<" : f32\n}\n";
    }
    s<<"%dense_vector = vector.from_elements ";for(int j=0;j<8;++j)s<<(j?", ":"")<<"%dense"<<j;
    s<<" : vector<8xf32>\nscf.yield %dense_vector : vector<8xf32>\n}\nscf.yield %selected, %yes : vector<8xf32>, i1\n}\n";
  } else s<<"scf.yield %result, "<<valid<<" : vector<8xf32>, i1\n}\n";
  v("orowbase","arith.addi %row0, %half8 : index");
  for(int j=0;j<8;++j) {
    s<<"%orow"<<j<<" = arith.addi %orowbase, %c"<<j<<" : index\n%oi"<<j<<"a = arith.muli %orow"<<j<<", %nn : index\n%oi"<<j<<" = arith.addi %oi"<<j<<"a, %bcol : index\n%o"<<j<<" = vector.extract %loop#0["<<j<<"] : f32 from vector<8xf32>\n";
    if(out!="f32")s<<"%cast"<<j<<" = arith.truncf %o"<<j<<" : f32 to "<<out<<"\n";
    s<<"memref.store %"<<(out=="f32"?"o":"cast")<<j<<", %out[%oi"<<j<<"] : memref<"<<m*n<<"x"<<out<<">\n";
  }
  v("si0","arith.muli %tile, %c32 : index");v("si","arith.addi %si0, %lane : index");v("status_value","arith.select %loop#1, %one32, %z32 : i32");
  s<<"memref.store %status_value, %status[%si] : memref<"<<(m/16)*(n/16)*32<<"xi32>\ngpu.return\n}}}\n";
  return text;
}
static LogicalResult scheduleNativeSparse(ModuleOp mod) {
  auto text=sparseGraphKernel(mod);if(failed(text))return failure();
  auto result=parseSourceString<ModuleOp>(*text,mod.getContext());
  if(!result)return mod.emitError("native sparse kernel failed verification");
  auto fn=*mod.getOps<func::FuncOp>().begin();
  auto a=cast<RankedTensorType>(fn.getArgument(0).getType());
  auto b=cast<RankedTensorType>(fn.getArgument(1).getType());
  auto out=cast<RankedTensorType>(fn.getResultTypes()[0]);
  Builder builder(mod.getContext());
  result->getOperation()->setAttr("tessera.sparse_selection",mod->getAttr("tessera.sparse_policy"));
  result->getOperation()->setAttr("tessera.sparse_shape", builder.getDenseI64ArrayAttr({a.getDimSize(0),b.getDimSize(1),a.getDimSize(1)}));
  result->getOperation()->setAttr("tessera.sparse_storage",builder.getStringAttr(a.getElementType().isF16()?"f16":"bf16"));
  result->getOperation()->setAttr("tessera.sparse_output",builder.getStringAttr(out.getElementType().isF32()?"f32":out.getElementType().isF16()?"f16":"bf16"));
  // Preserve the logical AD request as provenance; this pass emits only the
  // primal specialization. Reverse generation still consumes the original IR.
  for (StringRef name : {"tessera.autodiff", "tessera.autodiff.wrt", "tessera.autodiff.wrt_indices"})
    if (auto attr=mod->getAttr(name)) result->getOperation()->setAttr(name,attr);
  mod.getBodyRegion().takeBody(result->getBodyRegion());
  mod->setAttrs(result->getOperation()->getAttrs());
  return success();
}
} // namespace
