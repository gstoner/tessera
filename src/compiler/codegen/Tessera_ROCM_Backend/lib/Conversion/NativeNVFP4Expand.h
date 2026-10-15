// Native source-preserving NVFP4 expansion. One thread owns one K16 group.
// The checked launch boundary validates finite nonnegative E4M3 scales,
// positive finite projection globals, and representable rounded f32 products.
static mlir::LogicalResult emitNativeNVFP4Expand(mlir::ModuleOp module,
                                                mlir::Operation *op) {
  using namespace mlir;
  auto arch = module->getAttrOfType<StringAttr>("tessera.arch");
  if (!arch || arch.getValue() != "gfx1201")
    return op->emitError("native NVFP4 expansion requires an exact gfx1201 module");
  int64_t n = op->getAttrOfType<IntegerAttr>("n").getInt();
  int64_t k = op->getAttrOfType<IntegerAttr>("k").getInt();
  int64_t groups = k / 16;
  auto offsets = op->getAttrOfType<ArrayAttr>("row_offsets");
  std::string name = op->getAttrOfType<StringAttr>("name").getValue().str();
  OpBuilder b(module.getBodyRegion());
  b.setInsertionPointToEnd(module.getBody());
  Location l = op->getLoc();
  auto gm = b.create<gpu::GPUModuleOp>(l,name + "_mod");
  b.setInsertionPointToStart(&gm.getBodyRegion().front());
  auto bytes = MemRefType::get({ShapedType::kDynamic},b.getI8Type());
  auto doubles = MemRefType::get({ShapedType::kDynamic},b.getF64Type());
  auto floats = MemRefType::get({ShapedType::kDynamic},b.getF32Type());
  auto f = b.create<gpu::GPUFuncOp>(
      l,name,b.getFunctionType({bytes,bytes,doubles,bytes,floats},{}));
  f.setKernelAttr(b.getUnitAttr());
  f->setAttr("gpu.known_block_size",b.getDenseI32ArrayAttr({256,1,1}));
  b.setInsertionPointToStart(&f.getBody().front());
  auto ci = [&](int64_t v) -> Value { return b.create<arith::ConstantIndexOp>(l,v); };
  auto c32 = [&](int64_t v) -> Value { return b.create<arith::ConstantIntOp>(l,v,32); };
  auto c64 = [&](int64_t v) -> Value { return b.create<arith::ConstantIntOp>(l,v,64); };
  auto choose = [&](Value p,Value a,Value v) -> Value {
    return b.create<arith::SelectOp>(l,p,a,v);
  };
  auto eq = [&](Value a,Value v) -> Value {
    return b.create<arith::CmpIOp>(l,arith::CmpIPredicate::eq,a,v);
  };
  Value block = b.create<gpu::BlockIdOp>(l,gpu::Dimension::x);
  Value thread = b.create<gpu::ThreadIdOp>(l,gpu::Dimension::x);
  Value index = b.create<arith::AddIOp>(
      l,b.create<arith::MulIOp>(l,block,ci(256)),thread);
  Value active = b.create<arith::CmpIOp>(
      l,arith::CmpIPredicate::ult,index,ci(n*groups));
  auto guard = b.create<scf::IfOp>(l,active,false);
  b.setInsertionPointToStart(guard.thenBlock());
  Value row = b.create<arith::DivUIOp>(l,index,ci(groups));
  Value group = b.create<arith::RemUIOp>(l,index,ci(groups));
  Value projection = ci(0);
  for (unsigned p=1; p+1<offsets.size(); ++p) {
    Value after = b.create<arith::CmpIOp>(
        l,arith::CmpIPredicate::uge,row,ci(cast<IntegerAttr>(offsets[p]).getInt()));
    projection = choose(after,ci(p),projection);
  }
  Value global = b.create<memref::LoadOp>(l,f.getArgument(2),ValueRange{projection});
  Value raw = b.create<memref::LoadOp>(l,f.getArgument(1),ValueRange{index});
  Value bits = b.create<arith::ExtUIOp>(l,b.getI32Type(),raw);
  Value exponent = b.create<arith::AndIOp>(
      l,b.create<arith::ShRUIOp>(l,bits,c32(3)),c32(15));
  Value fraction = b.create<arith::AndIOp>(l,bits,c32(7));
  Value wideExp = b.create<arith::ExtUIOp>(l,b.getI64Type(),exponent);
  Value powBits = b.create<arith::ShLIOp>(
      l,b.create<arith::AddIOp>(l,wideExp,c64(1013)),c64(52));
  Value pow = b.create<arith::BitcastOp>(l,b.getF64Type(),powBits);
  Value normal = b.create<arith::MulFOp>(
      l,b.create<arith::UIToFPOp>(l,b.getF64Type(),
        b.create<arith::AddIOp>(l,fraction,c32(8))),pow);
  Value subPow = b.create<arith::ConstantOp>(l,b.getF64Type(),b.getF64FloatAttr(0x1p-9));
  Value subnormal = b.create<arith::MulFOp>(
      l,b.create<arith::UIToFPOp>(l,b.getF64Type(),fraction),subPow);
  Value magnitude = choose(eq(exponent,c32(0)),subnormal,normal);
  Value negative = b.create<arith::CmpIOp>(
      l,arith::CmpIPredicate::ne,b.create<arith::AndIOp>(l,bits,c32(128)),c32(0));
  Value signedScale = choose(negative,b.create<arith::NegFOp>(l,magnitude),magnitude);
  Value product = b.create<arith::MulFOp>(l,signedScale,global);
  Value rounded = b.create<arith::TruncFOp>(l,b.getF32Type(),product);
  Value scaleIndex = b.create<arith::AddIOp>(
      l,b.create<arith::MulIOp>(l,group,ci(n)),row);
  b.create<memref::StoreOp>(l,rounded,f.getArgument(4),ValueRange{scaleIndex});
  Value packedBase = b.create<arith::MulIOp>(l,index,ci(8));
  Value valueBase = b.create<arith::MulIOp>(l,index,ci(16));
  for (int j=0; j<8; ++j) {
    Value address = b.create<arith::AddIOp>(l,packedBase,ci(j));
    Value packed = b.create<arith::ExtUIOp>(
        l,b.getI32Type(),b.create<memref::LoadOp>(l,f.getArgument(0),ValueRange{address}));
    for (int half=0; half<2; ++half) {
      Value code = b.create<arith::AndIOp>(
          l,b.create<arith::ShRUIOp>(l,packed,c32(4*half)),c32(15));
      Value mag = b.create<arith::AndIOp>(l,code,c32(7));
      Value sign = b.create<arith::ShLIOp>(
          l,b.create<arith::AndIOp>(l,code,c32(8)),c32(4));
      Value expanded = b.create<arith::ShLIOp>(
          l,b.create<arith::AddIOp>(l,mag,c32(12)),c32(2));
      expanded = choose(eq(mag,c32(1)),c32(48),expanded);
      expanded = choose(eq(mag,c32(0)),c32(0),expanded);
      Value out = b.create<arith::TruncIOp>(
          l,b.getI8Type(),b.create<arith::OrIOp>(l,expanded,sign));
      Value destination = b.create<arith::AddIOp>(l,valueBase,ci(j*2+half));
      b.create<memref::StoreOp>(l,out,f.getArgument(3),ValueRange{destination});
    }
  }
  b.setInsertionPointAfter(guard);
  b.create<gpu::ReturnOp>(l);
  module->setAttr(gpu::GPUDialect::getContainerModuleAttrName(),b.getUnitAttr());
  return success();
}
