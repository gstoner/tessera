// Shared scalar oracle and opt-in cooperative SSD GPU lowering.
// Included inside namespace tessera after the common MLIR headers.
namespace {
static LogicalResult lowerNativeSSD(ModuleOp module) {
  SmallVector<schedule::SSDOp> work;
  module.walk([&](schedule::SSDOp op) { work.push_back(op); });
  for (auto op : work) {
    if (failed(op.verify())) return failure();
    OpBuilder b(op);
    Location loc = op.getLoc();
    auto xType = cast<RankedTensorType>(op.getX().getType());
    auto bType = cast<RankedTensorType>(op.getB().getType());
    auto index = [&](int64_t v) -> Value {
      return b.create<arith::ConstantIndexOp>(loc, v);
    };
    Value zero = index(0), one = index(1);
    Value time = index(xType.getDimSize(0)), heads = index(xType.getDimSize(1));
    Value width = index(xType.getDimSize(2)), states = index(bType.getDimSize(2));
    Value chunk = index(op.getChunkSize());
    Value fzero = b.create<arith::ConstantFloatOp>(loc, b.getF32Type(), APFloat(0.0f));
    auto empty = [&](Value result) -> Value {
      auto type = cast<RankedTensorType>(result.getType());
      return b.create<tensor::EmptyOp>(loc, type.getShape(), type.getElementType());
    };
    Value output = empty(op.getOutput()), checkpoints = empty(op.getCheckpoints());
    auto loop = [&](OpBuilder &builder, Value upper, ValueRange initial, auto body) {
      auto forOp = builder.create<scf::ForOp>(loc, zero, upper, one, initial,
          [&](OpBuilder &nested, Location at, Value iv, ValueRange args) {
            auto values = body(nested, iv, args);
            nested.create<scf::YieldOp>(at, values);
          });
      return SmallVector<Value>(forOp.getResults());
    };
    // Every Y element is written once, every checkpoint element at least once.
    // Rewriting a chunk's checkpoint at each step preserves the last state even
    // for a tail chunk, without allocating a full per-time residual tape.
    auto results = loop(b, time, ValueRange{op.getInitial(), output, checkpoints},
      [&](OpBuilder &tb, Value t, ValueRange ta) {
        Value ci = tb.create<arith::DivUIOp>(loc, t, chunk);
        return loop(tb, heads, ta, [&](OpBuilder &hb, Value h, ValueRange ha) {
          Value decay = hb.create<tensor::ExtractOp>(loc, op.getDecay(), ValueRange{t,h});
          return loop(hb, width, ha, [&](OpBuilder &pb, Value p, ValueRange pa) {
            Value x = pb.create<tensor::ExtractOp>(loc, op.getX(), ValueRange{t,h,p});
            auto ns = loop(pb, states, ValueRange{pa[0], pa[2], fzero},
              [&](OpBuilder &nb, Value n, ValueRange na) {
                Value old = nb.create<tensor::ExtractOp>(loc, na[0], ValueRange{h,n,p});
                Value bv = nb.create<tensor::ExtractOp>(loc, op.getB(), ValueRange{t,h,n});
                Value cv = nb.create<tensor::ExtractOp>(loc, op.getC(), ValueRange{t,h,n});
                Value carried = nb.create<arith::MulFOp>(loc, decay, old);
                Value added = nb.create<arith::MulFOp>(loc, bv, x);
                Value next = nb.create<arith::AddFOp>(loc, carried, added);
                Value state = nb.create<tensor::InsertOp>(loc, next, na[0], ValueRange{h,n,p});
                Value saved = nb.create<tensor::InsertOp>(loc, next, na[1], ValueRange{ci,h,n,p});
                Value weighted = nb.create<arith::MulFOp>(loc, cv, next);
                Value sum = nb.create<arith::AddFOp>(loc, na[2], weighted);
                return SmallVector<Value>{state, saved, sum};
              });
            Value y = pb.create<tensor::InsertOp>(loc, ns[2], pa[1], ValueRange{t,h,p});
            return SmallVector<Value>{ns[0], y, ns[1]};
          });
        });
      });
    op.getOutput().replaceAllUsesWith(results[1]);
    op.getCarry().replaceAllUsesWith(results[0]);
    op.getCheckpoints().replaceAllUsesWith(results[2]);
    op.erase();
  }
  return success();
}
} // namespace

namespace {
// One block owns (head, value-column); lanes own recurrent state elements.
// The reduction stays in state-index order, preserving the scalar oracle.
static LogicalResult lowerCooperativeSSD(ModuleOp module, StringRef backend) {
  if (backend != "nvidia" && backend != "rocm")
    return module.emitError("SSD GPU backend must be nvidia or rocm");
  auto functions = llvm::to_vector(module.getOps<func::FuncOp>());
  if (functions.size()!=1 || std::distance(module.getBody()->begin(),module.getBody()->end())!=1 ||
      functions[0].getBody().getBlocks().size()!=1)
    return module.emitError("cooperative SSD requires one isolated function");
  auto f=functions[0];
  auto ops=llvm::to_vector(f.getOps<schedule::SSDOp>());
  if (ops.size()!=1 || failed(ops[0].verify()) || f.getNumArguments()!=5 ||
      std::distance(f.front().begin(),f.front().end())!=2)
    return module.emitError("cooperative SSD requires the canonical Schedule entry");
  auto op=ops[0];
  for (unsigned i=0;i<5;++i) if (op->getOperand(i)!=f.getArgument(i))
    return module.emitError("SSD argument order disagrees");
  auto ret=dyn_cast<func::ReturnOp>(f.front().getTerminator());
  if (!ret || ret.getOperands()!=op.getResults())
    return module.emitError("SSD result order disagrees");
  auto x=cast<RankedTensorType>(op.getX().getType());
  auto bc=cast<RankedTensorType>(op.getB().getType());
  int64_t T=x.getDimSize(0), H=x.getDimSize(1), P=x.getDimSize(2), N=bc.getDimSize(2);
  if (N>256) return module.emitError("cooperative SSD supports at most 256 states");
  int64_t lanes=1; while (lanes<N) lanes*=2;
  std::string source; llvm::raw_string_ostream original(source); module.print(original); original.flush();
  std::string text; llvm::raw_string_ostream s(text);
  s<<"module attributes {tessera.ssd.source = "; StringAttr::get(module.getContext(),source).print(s);
  s<<", tessera.ssd.cooperative = true, tessera.autodiff.temporary_bytes = "<<lanes*4<<" : i64} { gpu.module @native_tape {\n";
  s<<"llvm.mlir.global private @ssd_reduction() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<"<<lanes<<" x f32>\n";
  s<<"gpu.func @product(";
  for (int i=0;i<8;++i) s<<"%p"<<i<<": !llvm.ptr<1>, ";
  s<<"%scratch: index) kernel attributes {known_block_size = array<i32: "<<lanes<<", 1, 1>} {\n";
  s<<"%marker = memref.alloca(%scratch) : memref<?xf32>\n\"tile.alloc_shared\"(%marker) : (memref<?xf32>) -> ()\n";
  s<<"%shared = llvm.mlir.addressof @ssd_reduction : !llvm.ptr<3>\n";
  for (auto pair : {std::pair<const char*,int64_t>{"T",T},{"H",H},{"P",P},{"N",N},{"chunk",op.getChunkSize()},{"zero",0},{"one",1}})
    s<<"%"<<pair.first<<" = arith.constant "<<pair.second<<" : index\n";
  s<<"%fz = arith.constant 0.0 : f32\n%lane = gpu.thread_id x\n%block = gpu.block_id x\n";
  s<<"%h = arith.divui %block, %P : index\n%p = arith.remui %block, %P : index\n%active = arith.cmpi ult, %lane, %N : index\n";
  s<<"%hn = arith.muli %h, %N : index\n%hnn = arith.addi %hn, %lane : index\n%base = arith.muli %hnn, %P : index\n%stateidx = arith.addi %base, %p : index\n";
  auto load=[&](StringRef name,int arg,StringRef idx) {
    s<<"%"<<name<<"i = arith.index_cast %"<<idx<<" : index to i64\n%"<<name<<"p = llvm.getelementptr %p"<<arg<<"[%"<<name<<"i] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32\n%"<<name<<" = llvm.load %"<<name<<"p : !llvm.ptr<1> -> f32\n";
  };
  auto store=[&](StringRef name,int arg,StringRef idx,StringRef val) {
    s<<"%"<<name<<"i = arith.index_cast %"<<idx<<" : index to i64\n%"<<name<<"p = llvm.getelementptr %p"<<arg<<"[%"<<name<<"i] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32\nllvm.store %"<<val<<", %"<<name<<"p : f32, !llvm.ptr<1>\n";
  };
  s<<"%initial = scf.if %active -> f32 {\n"; load("init",4,"stateidx");
  s<<"scf.yield %init : f32\n} else {scf.yield %fz : f32}\n";
  s<<"%final = scf.for %t = %zero to %T step %one iter_args(%state = %initial) -> f32 {\n";
  s<<"%th = arith.muli %t, %H : index\n%thh = arith.addi %th, %h : index\n%thp = arith.muli %thh, %P : index\n%xi = arith.addi %thp, %p : index\n";
  s<<"%next = scf.if %active -> f32 {\n";
  load("decay",1,"thh");load("xval",0,"xi");
  s<<"%thn = arith.muli %thh, %N : index\n%bci = arith.addi %thn, %lane : index\n";
  load("b",2,"bci");load("cval",3,"bci");
  s<<"%old = arith.mulf %decay, %state : f32\n%bx = arith.mulf %b, %xval : f32\n%new = arith.addf %old, %bx : f32\n%weighted = arith.mulf %cval, %new : f32\n";
  s<<"%li = arith.index_cast %lane : index to i64\n%lp = llvm.getelementptr %shared[0, %li] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<"<<lanes<<" x f32>\nllvm.store %weighted, %lp : f32, !llvm.ptr<3>\n";
  store("carry",6,"stateidx","new");
  s<<"%ci = arith.divui %t, %chunk : index\n%ch = arith.muli %ci, %H : index\n%chn = arith.muli %ch, %N : index\n%chnp = arith.muli %chn, %P : index\n%savedidx = arith.addi %chnp, %stateidx : index\n";
  store("saved",7,"savedidx","new");
  s<<"scf.yield %new : f32\n} else {\nscf.yield %state : f32\n}\ngpu.barrier\n";
  s<<"%leader = arith.cmpi eq, %lane, %zero : index\nscf.if %leader {\n%sum = scf.for %n = %zero to %N step %one iter_args(%acc = %fz) -> f32 {\n%ni = arith.index_cast %n : index to i64\n%np = llvm.getelementptr %shared[0, %ni] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<"<<lanes<<" x f32>\n%v = llvm.load %np : !llvm.ptr<3> -> f32\n%add = arith.addf %acc, %v : f32\nscf.yield %add : f32\n}\n";
  store("output",5,"xi","sum");
  s<<"}\ngpu.barrier\nscf.yield %next : f32\n}\ngpu.return\n} } }";s.flush();
  auto lowered=parseSourceString<ModuleOp>(text,module.getContext());
  if (!lowered) return failure();
  module->setAttrs((*lowered)->getAttrs());
  module.getBody()->clear();
  module.getBody()->getOperations().splice(module.getBody()->end(),lowered->getBody()->getOperations());
  return success();
}
} // namespace
