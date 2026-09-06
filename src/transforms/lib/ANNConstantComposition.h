// MSW-9 inference-only affine composition on native MLIR constants.
// Included by CanonicalizeTesseraIR.cpp inside its anonymous namespace.
struct ComposeConstantANN : public RewritePattern {
  ComposeConstantANN(MLIRContext *ctx) : RewritePattern("tessera.add", 3, ctx) {}
  LogicalResult matchAndRewrite(Operation *last, PatternRewriter &r) const override {
    auto plain = [](Operation *op, StringRef name) {
      if (!isOp(op,name) || op->getNumOperands()!=2 || op->getNumResults()!=1 || op->getNumRegions()!=0)
        return false;
      for (auto attr:op->getAttrs()) {
        auto flag=dyn_cast<BoolAttr>(attr.getValue());
        if (name!="tessera.matmul" || (attr.getName()!="transposeA" && attr.getName()!="transposeB") ||
            !flag || flag.getValue()) return false;
      }
      return true;
    };
    if (!plain(last,"tessera.add")) return failure();
    for (Operation *parent=last->getParentOp();parent;parent=parent->getParentOp())
      for (auto attr:parent->getAttrs())
        if (attr.getName().strref().contains("numeric_policy") || attr.getName().strref().contains("math_mode"))
          return failure();
    auto second=last->getOperand(0).getDefiningOp();
    if (!plain(second,"tessera.matmul") || !second->hasOneUse()) return failure();
    auto bias=second->getOperand(0).getDefiningOp();
    if (!plain(bias,"tessera.add") || !bias->hasOneUse()) return failure();
    auto first=bias->getOperand(0).getDefiningOp();
    if (!plain(first,"tessera.matmul") || !first->hasOneUse()) return failure();
    auto constant = [](Value v) -> DenseFPElementsAttr {
      auto c=v.getDefiningOp<arith::ConstantOp>();
      auto t=dyn_cast<RankedTensorType>(v.getType());
      if (!c || !t || t.getEncoding() || !t.hasStaticShape() || !t.getElementType().isF32()) return {};
      auto dense=dyn_cast<DenseFPElementsAttr>(c.getValue());
      if (!dense || llvm::any_of(dense.getValues<APFloat>(),[](const APFloat &v){ return !v.isFinite(); })) return {};
      return dense;
    };
    auto w1=constant(first->getOperand(1)), b1=constant(bias->getOperand(1));
    auto w2=constant(second->getOperand(1)), b2=constant(last->getOperand(1));
    if (!w1 || !b1 || !w2 || !b2) return failure();
    auto a=dyn_cast<RankedTensorType>(first->getOperand(0).getType());
    auto t1=cast<RankedTensorType>(w1.getType()), t2=cast<RankedTensorType>(w2.getType());
    if (!a || a.getEncoding() || !a.hasStaticShape() || !a.getElementType().isF32() ||
        a.getRank()!=2 || t1.getRank()!=2 || t2.getRank()!=2 ||
        b1.getType().getRank()!=2 || b2.getType().getRank()!=2) return failure();
    int64_t m=a.getDimSize(0), k=t1.getDimSize(0), h=t1.getDimSize(1), n=t2.getDimSize(1);
    if (m<=0 || k<=0 || h<=0 || n<=0 || a.getDimSize(1)!=k || t2.getDimSize(0)!=h ||
        b1.getType().getShape()!=ArrayRef<int64_t>({m,h}) || b2.getType().getShape()!=ArrayRef<int64_t>({m,n})) return failure();
    // Bound compile-time work and constant expansion without overflow.
    if (m>1024 || k>1024 || h>1024 || n>1024 || (k+m)*h*n>1048576) return failure();
    auto tensor=[&](ArrayRef<int64_t> shape){return RankedTensorType::get(shape,r.getF32Type());};
    if (first->getResult(0).getType()!=tensor({m,h}) || bias->getResult(0).getType()!=tensor({m,h}) ||
        second->getResult(0).getType()!=tensor({m,n}) || last->getResult(0).getType()!=tensor({m,n})) return failure();
    SmallVector<APFloat> x(w1.getValues<APFloat>()), y(w2.getValues<APFloat>());
    SmallVector<APFloat> p(b1.getValues<APFloat>()), q(b2.getValues<APFloat>()), weights, biases;
    auto dot=[&](int64_t row,int64_t col,bool biasRow){
      APFloat sum(0.0f);
      for (int64_t i=0;i<h;++i) {
        APFloat product=biasRow ? p[row*h+i] : x[row*h+i];
        product.multiply(y[i*n+col],APFloat::rmNearestTiesToEven);
        sum.add(product,APFloat::rmNearestTiesToEven);
      }
      if (biasRow) sum.add(q[row*n+col],APFloat::rmNearestTiesToEven);
      return sum;
    };
    for (int64_t i=0;i<k;++i) for (int64_t j=0;j<n;++j) weights.push_back(dot(i,j,false));
    for (int64_t i=0;i<m;++i) for (int64_t j=0;j<n;++j) biases.push_back(dot(i,j,true));
    if (llvm::any_of(weights,[](const APFloat &v){return !v.isFinite();}) ||
        llvm::any_of(biases,[](const APFloat &v){return !v.isFinite();})) return failure();
    auto weight=r.create<arith::ConstantOp>(last->getLoc(),DenseFPElementsAttr::get(tensor({k,n}),weights));
    auto offset=r.create<arith::ConstantOp>(last->getLoc(),DenseFPElementsAttr::get(tensor({m,n}),biases));
    OperationState mm(last->getLoc(),"tessera.matmul");
    mm.addOperands({first->getOperand(0),weight.getResult()}); mm.addTypes(tensor({m,n}));
    auto fused=r.create(mm);
    OperationState add(last->getLoc(),"tessera.add");
    add.addOperands({fused->getResult(0),offset.getResult()}); add.addTypes(tensor({m,n}));
    r.replaceOp(last,r.create(add)->getResults());
    return success();
  }
};
