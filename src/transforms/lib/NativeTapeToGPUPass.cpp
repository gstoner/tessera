// Split, bufferized AD products to a bounded serial native GPU entry.
#include "Tessera/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>
#include <algorithm>
namespace tessera {
namespace {
struct NativeTapeToGPUPass : mlir::PassWrapper<NativeTapeToGPUPass, mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NativeTapeToGPUPass)
  NativeTapeToGPUPass() = default;
  NativeTapeToGPUPass(const NativeTapeToGPUPass &other) : PassWrapper(other) {}
  Option<std::string> backend{*this, "backend", llvm::cl::desc("Private allocation lowering: nvidia or rocm"), llvm::cl::init("nvidia")};
  Option<bool> parallelRows{*this, "parallel-ann-rows", llvm::cl::desc("Assign proven independent ANN rows to GPU threads"), llvm::cl::init(false)};
  Option<bool> statusBuffer{*this, "status-buffer", llvm::cl::desc("Append a checked i64 status result for top-level product assertions"), llvm::cl::init(false)};
  llvm::StringRef getArgument() const final { return "tessera-native-tape-to-gpu"; }
  llvm::StringRef getDescription() const final { return "Materialize bounded bufferized AD/ANN products with proved temporary capacities"; }
  void getDependentDialects(mlir::DialectRegistry &r) const override {
    r.insert<mlir::arith::ArithDialect,mlir::func::FuncDialect,mlir::gpu::GPUDialect,
      mlir::LLVM::LLVMDialect,mlir::memref::MemRefDialect,mlir::scf::SCFDialect,mlir::math::MathDialect,mlir::cf::ControlFlowDialect>();
  }
  void runOnOperation() override {
    using namespace mlir;
    auto m=getOperation();
    auto reject=[&]() { m.emitError("native tape GPU requires an isolated static f32/f64 buffer product with bounded for/if control and at most 4096 temporary bytes"); signalPassFailure(); };
    if (backend!="nvidia" && backend!="rocm") return reject();
    SmallVector<func::FuncOp> fs(m.getOps<func::FuncOp>());
    bool ad=m->hasAttr("tessera.autodiff.product_abi") && m->hasAttr("tessera.autodiff.product_pair");
    bool ann=m->hasAttr("tessera.ann.source");
    if (fs.size()!=1 || (!ad && !ann) || (ad && ann)) return reject();
    auto f=fs[0];
    if (!f.getBody().hasOneBlock() || f.getNumResults() || !isa<func::ReturnOp>(f.getBody().front().getTerminator())) return reject();
    if (statusBuffer) {
      if (!ad || parallelRows) return reject();
      // Only top-level guards can be turned into a guarded suffix without
      // changing loop/region semantics. Never erase a nested assertion.
      SmallVector<cf::AssertOp> guards;
      bool nested=false;
      f.walk([&](cf::AssertOp guard) {
        if (guard->getBlock()!=&f.getBody().front()) nested=true;
        guards.push_back(guard);
      });
      if (nested) return reject();
      OpBuilder entry=OpBuilder::atBlockBegin(&f.getBody().front());
      auto statusType=MemRefType::get({1},entry.getI64Type());
      if (failed(f.insertArgument(f.getNumArguments(),statusType,DictionaryAttr{},f.getLoc()))) return reject();
      auto status=f.getArguments().back();
      auto zero=arith::ConstantIndexOp::create(entry,f.getLoc(),0);
      auto success=arith::ConstantIntOp::create(entry,f.getLoc(),0,64);
      memref::StoreOp::create(entry,f.getLoc(),success,status,ValueRange{zero});
      // Work backwards so each later guard and suffix move together. On a
      // failure no following memory operation executes and status stays failed.
      for (auto guard:llvm::reverse(guards)) {
        OpBuilder at(guard);
        auto branch=scf::IfOp::create(at,guard.getLoc(),guard.getArg(),true);
        auto *end=f.getBody().front().getTerminator();
        auto *next=guard->getNextNode();
        while (next!=end) {
          auto *following=next->getNextNode();
          next->moveBefore(branch.thenBlock()->getTerminator());
          next=following;
        }
        at.setInsertionPoint(branch.elseBlock()->getTerminator());
        auto failure=arith::ConstantIntOp::create(at,guard.getLoc(),1,64);
        memref::StoreOp::create(at,guard.getLoc(),failure,status,ValueRange{zero});
        guard.erase();
      }
      m->setAttr("tessera.autodiff.gpu_status",entry.getStringAttr("guard-v1"));
    }
    auto admissible=[](Type t) {
      auto mt=dyn_cast<MemRefType>(t);
      if (!mt || !mt.hasStaticShape() || !(mt.getElementType().isF32() || mt.getElementType().isF64() || mt.getElementType().isInteger(8) || mt.getElementType().isInteger(64) || mt.getElementType().isInteger(1) || mt.getElementType().isIndex()) || !mt.getLayout().isIdentity() || mt.getMemorySpace()) return false;
      int64_t count=1;
      for (auto d:mt.getShape()) {
        if (d<=0 || d>1024 || count>1024/d) return false;
        count*=d;
      }
      return true;
    };
    auto elementBytes=[](Type t) -> int64_t { return t.isIndex() ? 8 : (t.getIntOrFloatBitWidth()+7)/8; };
    for (auto t:f.getArgumentTypes()) if (!admissible(t)) return reject();
    // Derive replay-loop capacity from SSA arithmetic over enclosing bounded
    // induction variables. Never trust a user-written max_iters attribute or
    // a loaded counter. Intervals stay small so endpoint arithmetic cannot wrap.
    using Interval = std::pair<int64_t,int64_t>;
    std::function<std::optional<Interval>(Value,unsigned)> range;
    range = [&](Value value,unsigned depth) -> std::optional<Interval> {
      if (depth>64 || !value.getType().isIndex()) return std::nullopt;
      APInt constant;
      if (matchPattern(value,m_ConstantInt(&constant))) {
        if (!constant.isSignedIntN(64)) return std::nullopt;
        auto v=constant.getSExtValue();
        if (v < -1024 || v > 1024) return std::nullopt;
        return Interval{v,v};
      }
      if (auto arg=dyn_cast<BlockArgument>(value)) {
        auto owner=dyn_cast_or_null<scf::ForOp>(arg.getOwner()->getParentOp());
        if (!owner || owner->hasAttr("unsignedCmp") || value!=owner.getInductionVar()) return std::nullopt;
        auto low=range(owner.getLowerBound(),depth+1);
        auto high=range(owner.getUpperBound(),depth+1);
        auto step=range(owner.getStep(),depth+1);
        if (!low || !high || !step || step->first<=0 || step->first!=step->second)
          return std::nullopt;
        return Interval{low->first,std::max(low->first,high->second-1)};
      }
      auto *op=value.getDefiningOp();
      if (!op) return std::nullopt;
      if (auto select=dyn_cast<arith::SelectOp>(op)) {
        auto a=range(select.getTrueValue(),depth+1), b=range(select.getFalseValue(),depth+1);
        if (!a || !b) return std::nullopt;
        return Interval{std::min(a->first,b->first),std::max(a->second,b->second)};
      }
      if (op->getNumOperands()!=2) return std::nullopt;
      auto a=range(op->getOperand(0),depth+1), b=range(op->getOperand(1),depth+1);
      if (!a || !b) return std::nullopt;
      Interval result;
      if (isa<arith::AddIOp>(op)) result={a->first+b->first,a->second+b->second};
      else if (isa<arith::SubIOp>(op)) result={a->first-b->second,a->second-b->first};
      else if (isa<arith::MulIOp>(op)) {
        int64_t products[]={a->first*b->first,a->first*b->second,a->second*b->first,a->second*b->second};
        result={*std::min_element(products,products+4),*std::max_element(products,products+4)};
      } else return std::nullopt;
      if (result.first < -1024 || result.second > 1024) return std::nullopt;
      return result;
    };
    auto tripCapacity=[&](scf::ForOp loop) -> std::optional<int64_t> {
      if (loop->hasAttr("unsignedCmp")) return std::nullopt;
      auto low=range(loop.getLowerBound(),0), high=range(loop.getUpperBound(),0), step=range(loop.getStep(),0);
      if (!low || !high || !step || low->first<0 ||
          step->first!=step->second || step->first<=0) return std::nullopt;
      auto count=std::max<int64_t>(1,(high->second-low->first+step->first-1)/step->first);
      if (count>1024) return std::nullopt;
      return count;
    };
    // Physical capacity is proved from SSA ranges; logical dimensions stay
    // dynamic in each view and copy. Loaded extents and unknown aliases refuse.
    auto allocationShape=[&](Operation *op) -> std::optional<SmallVector<int64_t>> {
      auto type=dyn_cast<MemRefType>(op->getResult(0).getType());
      if (!type) return std::nullopt;
      SmallVector<int64_t> shape(type.getShape());
      if (!type.hasStaticShape()) {
        auto alloc=dyn_cast<memref::AllocOp>(op);
        if (!alloc) return std::nullopt;
        unsigned index=0;
        for (auto &dim:shape) if (ShapedType::isDynamic(dim)) {
          auto bounds=range(alloc.getDynamicSizes()[index++],0);
          if (!bounds || bounds->first<0 || bounds->second<=0) return std::nullopt;
          dim=bounds->second;
        }
      }
      auto capacity=MemRefType::get(shape,type.getElementType(),type.getLayout(),type.getMemorySpace());
      if (!admissible(capacity)) return std::nullopt;
      return shape;
    };
    bool bad=false;
    int64_t bytes=0;
    f.walk([&](Operation *op) {
      auto dialect=op->getName().getDialectNamespace();
      if (op!=f.getOperation() && dialect!="arith" && dialect!="math" && dialect!="memref" && dialect!="scf" && !isa<func::ReturnOp>(op)) bad=true;
      if (isa<scf::WhileOp,func::CallOp,memref::DeallocOp>(op)) bad=true;
      if (auto copy=dyn_cast<memref::CopyOp>(op)) {
        auto source=cast<MemRefType>(copy.getSource().getType());
        auto target=cast<MemRefType>(copy.getTarget().getType());
        // A dynamic copy must have identical logical shape SSA on both
        // allocations; capacity equality is not a logical shape proof.
        if (source.getShape()!=target.getShape()) bad=true;
        if (!source.hasStaticShape() || !target.hasStaticShape()) {
          auto a=copy.getSource().getDefiningOp<memref::AllocOp>();
          auto b=copy.getTarget().getDefiningOp<memref::AllocOp>();
          if (!a || !b || !llvm::equal(a.getDynamicSizes(),b.getDynamicSizes())) bad=true;
        }
      }
      if (op->getNumRegions() && op!=f.getOperation() && !isa<scf::ForOp,scf::IfOp>(op)) bad=true;
      if (auto loop=dyn_cast<scf::ForOp>(op); loop && !tripCapacity(loop)) bad=true;
      if (!isa<memref::AllocOp,memref::AllocaOp,memref::GetGlobalOp>(op)) return;
      auto capacity=allocationShape(op);
      if (!capacity) { bad=true; return; }
      int64_t n=elementBytes(cast<MemRefType>(op->getResult(0).getType()).getElementType());
      for (int64_t dim:*capacity) n*=dim;
      for (Operation *parent=op->getParentOp();parent && parent!=f.getOperation();parent=parent->getParentOp())
        if (auto loop=dyn_cast<scf::ForOp>(parent)) {
          auto capacity=tripCapacity(loop);
          if (!capacity || n>4096 / *capacity) { bad=true; return; }
          int64_t trip=*capacity;
          n*=trip;
        }
      bytes+=n;
      if (bytes>4096) bad=true;
      if (auto get=dyn_cast<memref::GetGlobalOp>(op)) {
        auto global=m.lookupSymbol<memref::GlobalOp>(get.getName());
        auto value=global ? dyn_cast_or_null<DenseElementsAttr>(global.getInitialValueAttr()) : DenseElementsAttr();
        if (!global || !global.getConstant() || !value) bad=true;
      }
    });
    int64_t rowCount=0;
    if (parallelRows) {
      if (!ann || f.getNumArguments()!=2) return reject();
      auto input=cast<MemRefType>(f.getArgument(0).getType());
      auto output=cast<MemRefType>(f.getArgument(1).getType());
      if (input.getRank()!=2 || output.getRank()<1 ||
          input.getDimSize(0)<2 || input.getDimSize(0)>64 ||
          output.getDimSize(0)!=input.getDimSize(0)) return reject();
      rowCount=input.getDimSize(0);
      auto rowLoop=[&](scf::ForOp loop) {
        APInt low,high,step;
        return loop.getNumRegionIterArgs()==0 &&
               matchPattern(loop.getLowerBound(),m_ConstantInt(&low)) && low.isZero() &&
               matchPattern(loop.getUpperBound(),m_ConstantInt(&high)) && high==rowCount &&
               matchPattern(loop.getStep(),m_ConstantInt(&step)) && step.isOne();
      };
      auto owned=[&](Value memref) {
        if (memref==f.getArgument(0) || memref==f.getArgument(1)) return true;
        return isa_and_nonnull<memref::AllocOp,memref::AllocaOp>(memref.getDefiningOp());
      };
      auto access=[&](Operation *op,Value memref,ValueRange indices,bool write) {
        if (memref.getDefiningOp<memref::GetGlobalOp>()) return !write;
        auto type=cast<MemRefType>(memref.getType());
        if (!owned(memref) || indices.empty() || type.getDimSize(0)!=rowCount ||
            (write && memref==f.getArgument(0))) return false;
        Operation *outer=op;
        while (outer->getParentOp()!=f.getOperation()) outer=outer->getParentOp();
        auto loop=dyn_cast<scf::ForOp>(outer);
        return loop && rowLoop(loop) && indices.front()==loop.getInductionVar();
      };
      f.walk([&](Operation *op) {
        // Memory operations not covered by the row-access proof (atomics,
        // alias construction, DMA, etc.) must not bypass it.
        if (op->getName().getDialectNamespace()=="memref" &&
            !isa<memref::AllocOp,memref::AllocaOp,memref::GetGlobalOp,
                 memref::LoadOp,memref::StoreOp,memref::CopyOp,memref::DimOp>(op)) bad=true;
        if (auto loop=dyn_cast<scf::ForOp>(op); loop && loop->getParentOp()==f.getOperation() && !rowLoop(loop)) bad=true;
        if (auto load=dyn_cast<memref::LoadOp>(op); load && !access(op,load.getMemRef(),load.getIndices(),false)) bad=true;
        if (auto store=dyn_cast<memref::StoreOp>(op); store && !access(op,store.getMemRef(),store.getIndices(),true)) bad=true;
        if (auto copy=dyn_cast<memref::CopyOp>(op)) {
          auto type=cast<MemRefType>(copy.getSource().getType());
          if (op->getParentOp()!=f.getOperation() || type.getRank()<1 || type.getDimSize(0)!=rowCount ||
              !owned(copy.getTarget()) || copy.getTarget()==f.getArgument(0) ||
              (!owned(copy.getSource()) && !copy.getSource().getDefiningOp<memref::GetGlobalOp>())) bad=true;
        }
      });
    }
    if (bad) return reject();
    std::string text; llvm::raw_string_ostream os(text);
    os<<"module { gpu.module @native_tape { gpu.func @product(";
    for (unsigned i=0;i<f.getNumArguments();++i) os<<"%p"<<i<<": !llvm.ptr<1>, ";
    os<<"%scratch: index) kernel {\n";
    for (auto [i,t]:llvm::enumerate(f.getArgumentTypes())) {
      auto mt=cast<MemRefType>(t); auto rank=mt.getRank();
      std::string desc; llvm::raw_string_ostream ds(desc);
      ds<<"!llvm.struct<(ptr, ptr, i64";
      if (rank) ds<<", array<"<<rank<<" x i64>, array<"<<rank<<" x i64>";
      ds<<")>"; ds.flush();
      os<<"%raw"<<i<<" = llvm.addrspacecast %p"<<i<<" : !llvm.ptr<1> to !llvm.ptr\n";
      os<<"%d"<<i<<"_0 = llvm.mlir.undef : "<<desc<<"\n";
      unsigned version=0;
      auto insert=[&](StringRef value,std::string position) {
        os<<"%d"<<i<<"_"<<(version+1)<<" = llvm.insertvalue "<<value<<", %d"<<i<<"_"<<version<<"["<<position<<"] : "<<desc<<"\n"; ++version;
      };
      insert("%raw"+std::to_string(i),"0"); insert("%raw"+std::to_string(i),"1");
      os<<"%z"<<i<<" = llvm.mlir.constant(0 : i64) : i64\n"; insert("%z"+std::to_string(i),"2");
      int64_t stride=1;
      for (int d=rank-1;d>=0;--d) {
        auto prefix="%shape"+std::to_string(i)+"_"+std::to_string(d);
        os<<prefix<<" = llvm.mlir.constant("<<mt.getDimSize(d)<<" : i64) : i64\n";
        insert(prefix,"3, "+std::to_string(d));
        os<<prefix<<"s = llvm.mlir.constant("<<stride<<" : i64) : i64\n";
        insert(prefix+"s","4, "+std::to_string(d)); stride*=mt.getDimSize(d);
      }
      os<<"%m"<<i<<" = builtin.unrealized_conversion_cast %d"<<i<<"_"<<version<<" : "<<desc<<" to "<<mt<<"\n";
    }
    os<<"%marker = memref.alloca(%scratch) : memref<?xf32>\n\"tile.alloc_shared\"(%marker) : (memref<?xf32>) -> ()\ngpu.return\n} } }"; os.flush();
    auto gpu=parseSourceString<ModuleOp>(text,m.getContext());
    if (!gpu) return reject();
    auto deviceModule=*gpu->getOps<gpu::GPUModuleOp>().begin();
    auto kernel=*deviceModule.getOps<gpu::GPUFuncOp>().begin();
    IRMapping mapping; unsigned argument=0;
    kernel.walk([&](UnrealizedConversionCastOp cast){mapping.map(f.getArgument(argument++),cast.getResult(0));});
    OpBuilder b(kernel.getBody().front().getTerminator());
    for (auto &op:f.getBody().front().without_terminator()) b.clone(op,mapping);
    SmallVector<Operation *> memory;
    kernel.walk([&](Operation *op){if (isa<memref::AllocOp,memref::GetGlobalOp,memref::CopyOp>(op) || (isa<memref::AllocaOp>(op) && cast<MemRefType>(op->getResult(0).getType()).hasStaticShape())) memory.push_back(op);});
    for (auto *op:memory) {
      OpBuilder at(op); auto loc=op->getLoc();
      auto loopNest=[&](MemRefType type,auto leaf) {
        SmallVector<Value> indices;
        std::function<void(int)> recurse=[&](int dim) {
          if (dim==type.getRank()) { leaf(at,indices); return; }
          auto zero=arith::ConstantIndexOp::create(at,loc,0), one=arith::ConstantIndexOp::create(at,loc,1);
          Value end;
          if (type.isDynamicDim(dim))
            end=memref::DimOp::create(at,loc,cast<memref::CopyOp>(op).getSource(),dim);
          else end=arith::ConstantIndexOp::create(at,loc,type.getDimSize(dim));
          auto loop=scf::ForOp::create(at,loc,zero,end,one);
          OpBuilder::InsertionGuard guard(at); at.setInsertionPointToStart(loop.getBody());
          indices.push_back(loop.getInductionVar()); recurse(dim+1); indices.pop_back();
        }; recurse(0);
      };
      if (auto copy=dyn_cast<memref::CopyOp>(op)) {
        loopNest(cast<MemRefType>(copy.getSource().getType()),[&](OpBuilder &builder,ValueRange indices) {
          auto v=memref::LoadOp::create(builder,loc,copy.getSource(),indices);
          memref::StoreOp::create(builder,loc,v,copy.getTarget(),indices);
        }); op->erase(); continue;
      }
      auto type=cast<MemRefType>(op->getResult(0).getType());
      // A GPU static alloca is not a fresh allocation on each loop trip.
      // Give every syntactic allocation an entry-owned byte array and derive
      // a distinct slot from the complete enclosing iteration path.
      SmallVector<scf::ForOp> loops;
      int64_t slots=1;
      for (auto *parent=op->getParentOp();parent!=kernel.getOperation();parent=parent->getParentOp())
        if (auto loop=dyn_cast<scf::ForOp>(parent)) loops.push_back(loop);
      Value slot=arith::ConstantIndexOp::create(at,loc,0);
      for (auto loop:llvm::reverse(loops)) {
        // Cloning preserves the validated SSA range relationships.
        int64_t trip=*tripCapacity(loop);
        slots*=trip;
        auto count=arith::ConstantIndexOp::create(at,loc,trip);
        auto delta=arith::SubIOp::create(at,loc,loop.getInductionVar(),loop.getLowerBound());
        auto ordinal=arith::DivUIOp::create(at,loc,delta,loop.getStep());
        slot=arith::AddIOp::create(at,loc,arith::MulIOp::create(at,loc,slot,count),ordinal);
      }
      auto capacity=allocationShape(op);
      if (!capacity) return reject();
      int64_t slotBytes=elementBytes(type.getElementType());
      for (int64_t dim:*capacity) slotBytes*=dim;
      SmallVector<Value> dynamicSizes;
      if (auto alloc=dyn_cast<memref::AllocOp>(op)) llvm::append_range(dynamicSizes,alloc.getDynamicSizes());
      OpBuilder entry=OpBuilder::atBlockBegin(&kernel.getBody().front());
      // AMDGPU requires explicit private FrameIndex addressing. NVVM owns
      // conversion of generic allocas to its local address representation;
      // forcing AMD's private descriptor representation through NVVM is unsafe.
      Attribute space=backend=="rocm" ? Attribute(entry.getI64IntegerAttr(5)) : Attribute();
      auto root=memref::AllocaOp::create(entry,loc,MemRefType::get({slots*slotBytes},entry.getI8Type(),MemRefLayoutAttrInterface{},space));
      root->setAttr("alignment",entry.getI64IntegerAttr(64));
      auto stride=arith::ConstantIndexOp::create(at,loc,slotBytes);
      auto offset=arith::MulIOp::create(at,loc,slot,stride);
      auto privateType=MemRefType::get(type.getShape(),type.getElementType(),MemRefLayoutAttrInterface{},space);
      auto view=memref::ViewOp::create(at,loc,privateType,root,offset,dynamicSizes);
      Value local=view;
      if (space) local=memref::MemorySpaceCastOp::create(at,loc,type,view);
      if (auto get=dyn_cast<memref::GetGlobalOp>(op)) {
        auto global=m.lookupSymbol<memref::GlobalOp>(get.getName());
        auto values=cast<DenseElementsAttr>(global.getInitialValueAttr());
        // Constants are bounded by the same 4096-byte resource check. Preserve
        // every element; ANN weights need not be splats.
        int64_t ordinal=0;
        for (auto value:values.getValues<Attribute>()) {
          SmallVector<Value> indices(type.getRank());
          int64_t remaining=ordinal++;
          for (int d=type.getRank()-1;d>=0;--d) {
            indices[d]=arith::ConstantIndexOp::create(at,loc,remaining%type.getDimSize(d));
            remaining/=type.getDimSize(d);
          }
          auto v=arith::ConstantOp::create(at,loc,type.getElementType(),cast<TypedAttr>(value));
          memref::StoreOp::create(at,loc,v,local,indices);
        }
      }
      op->getResult(0).replaceAllUsesWith(local); op->erase();
    }
    if (parallelRows) {
      OpBuilder at=OpBuilder::atBlockBegin(&kernel.getBody().front());
      auto loc=kernel.getLoc();
      Value row=gpu::ThreadIdOp::create(at,loc,gpu::Dimension::x);
      Value one=arith::ConstantIndexOp::create(at,loc,1);
      Value end=arith::AddIOp::create(at,loc,row,one);
      // Every mutable access was proved row-local before cloning. Constants
      // remain complete private/read-only arrays in every thread.
      for (auto loop:kernel.getBody().front().getOps<scf::ForOp>()) {
        loop.getLowerBoundMutable().assign(row);
        loop.getUpperBoundMutable().assign(end);
      }
      kernel->setAttr("known_block_size",at.getDenseI32ArrayAttr({static_cast<int32_t>(rowCount),1,1}));
    }
    m.getBody()->clear();
    m.getBody()->getOperations().splice(m.getBody()->end(),gpu->getBody()->getOperations());
    m->setAttr("tessera.autodiff.temporary_bytes",IntegerAttr::get(IntegerType::get(m.getContext(),64),bytes));
  }
};
}
std::unique_ptr<mlir::Pass> createNativeTapeToGPUPass() { return std::make_unique<NativeTapeToGPUPass>(); }
}
