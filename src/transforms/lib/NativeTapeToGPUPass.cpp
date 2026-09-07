// Split, bufferized AD products to a bounded serial native GPU entry.
#include "Tessera/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Math/IR/Math.h"
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
  llvm::StringRef getArgument() const final { return "tessera-native-tape-to-gpu"; }
  llvm::StringRef getDescription() const final { return "Materialize a bounded static bufferized AD product as a serial GPU entry"; }
  void getDependentDialects(mlir::DialectRegistry &r) const override {
    r.insert<mlir::arith::ArithDialect,mlir::func::FuncDialect,mlir::gpu::GPUDialect,
      mlir::LLVM::LLVMDialect,mlir::memref::MemRefDialect,mlir::scf::SCFDialect,mlir::math::MathDialect>();
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
    bool bad=false;
    int64_t bytes=0;
    f.walk([&](Operation *op) {
      auto dialect=op->getName().getDialectNamespace();
      if (op!=f.getOperation() && dialect!="arith" && dialect!="math" && dialect!="memref" && dialect!="scf" && !isa<func::ReturnOp>(op)) bad=true;
      if (isa<scf::WhileOp,func::CallOp,memref::DeallocOp>(op)) bad=true;
      if (auto copy=dyn_cast<memref::CopyOp>(op)) {
        auto source=cast<MemRefType>(copy.getSource().getType());
        auto target=cast<MemRefType>(copy.getTarget().getType());
        // The serial copy expansion uses compile-time bounds, never the
        // dynamic-dimension sentinel as a loop extent.
        if (!source.hasStaticShape() || !target.hasStaticShape() || source.getShape()!=target.getShape()) bad=true;
      }
      if (op->getNumRegions() && op!=f.getOperation() && !isa<scf::ForOp,scf::IfOp>(op)) bad=true;
      if (auto loop=dyn_cast<scf::ForOp>(op); loop && !tripCapacity(loop)) bad=true;
      if (!isa<memref::AllocOp,memref::AllocaOp,memref::GetGlobalOp>(op)) return;
      if (!admissible(op->getResult(0).getType())) { bad=true; return; }
      int64_t n=cast<MemRefType>(op->getResult(0).getType()).getNumElements()*
          elementBytes(cast<MemRefType>(op->getResult(0).getType()).getElementType());
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
          auto zero=arith::ConstantIndexOp::create(at,loc,0), one=arith::ConstantIndexOp::create(at,loc,1), end=arith::ConstantIndexOp::create(at,loc,type.getDimSize(dim));
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
      int64_t slotBytes=type.getNumElements()*elementBytes(type.getElementType());
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
      auto view=memref::ViewOp::create(at,loc,privateType,root,offset,ValueRange{});
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
    m.getBody()->clear();
    m.getBody()->getOperations().splice(m.getBody()->end(),gpu->getBody()->getOperations());
    m->setAttr("tessera.autodiff.temporary_bytes",IntegerAttr::get(IntegerType::get(m.getContext(),64),bytes));
  }
};
}
std::unique_ptr<mlir::Pass> createNativeTapeToGPUPass() { return std::make_unique<NativeTapeToGPUPass>(); }
}
