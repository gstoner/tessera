"""Independent CPU reference for typed, exact block-scaled products.

Used by eager frontend/differential checks; it is not a production backend.
"""
def reference_typed_scaled_matmul(a, b, scale_a, scale_b, *, numeric_policy,
                                 scale_layout, transposeA=False, transposeB=False,
                                 batching=None):
    import numpy as np
    if type(transposeA) is not bool or type(transposeB) is not bool:
        raise ValueError("scaled_matmul transpose attributes must be boolean")
    broadcast = batching == "broadcast"
    batched = batching is not None
    lhs_batched = batching in {"shared_rhs_rows","independent_rhs"}
    rhs_batched = batching in {"independent_rhs","shared_lhs"}
    if batching not in {None,"shared_rhs_rows","independent_rhs","shared_lhs","broadcast"}:
        raise ValueError("typed scaled reference batching policy is unsupported")
    if batched and not broadcast and transposeA:
        raise ValueError("typed shared-RHS reference requires no lhs transpose")
    if numeric_policy != {"accum":"fp32", "execution_mode":"exact_per_block"}:
        raise ValueError("typed scaled reference requires exact fp32 block accumulation")
    block = scale_layout.get("block") if isinstance(scale_layout,dict) else None
    if (not isinstance(scale_layout,dict) or set(scale_layout)!={"granularity","block","format"} or
        scale_layout["granularity"]!="block" or scale_layout["format"] not in {"fp32","e8m0"} or
        not isinstance(block,(list,tuple)) or len(block)!=2 or
        any(type(x) is not int or x<=0 for x in block)):
        raise ValueError("typed scaled reference requires explicit positive fp32/E8M0 block scales")
    encoded = scale_layout["format"] == "e8m0"
    if encoded and list(block) != [1,32]:
        raise ValueError("typed E8M0 reference requires block [1,32]")
    aa,bb,sa,sb = (np.asarray(v) for v in (a,b,scale_a,scale_b))
    invalid_rank = (aa.ndim < 2 or bb.ndim < 2) if broadcast else (
        (aa.ndim<3 if lhs_batched else aa.ndim!=2) or
        (bb.ndim<3 if rhs_batched else bb.ndim!=2))
    if invalid_rank:
        raise ValueError("typed scaled reference requires matrix operands")
    if str(aa.dtype)!="float8_e4m3fn" or str(bb.dtype)!="float8_e4m3fn":
        raise ValueError("typed scaled reference requires rank-two E4M3FN arrays")
    scale_dtype = np.dtype(np.uint8 if encoded else np.float32)
    if sa.dtype!=scale_dtype or sb.dtype!=scale_dtype:
        raise ValueError("typed scaled reference scale storage differs from declared format")
    aa=aa.swapaxes(-1,-2) if transposeA else aa
    bb=bb.swapaxes(-1,-2) if transposeB else bb
    m,k=aa.shape[-2:];kb,n=bb.shape[-2:]
    batch_shape=aa.shape[:-2] if lhs_batched else bb.shape[:-2] if rhs_batched else ()
    lhs_batch_shape=aa.shape[:-2];rhs_batch_shape=bb.shape[:-2]
    scale_n,scale_k=block
    groups=(k+scale_k-1)//scale_k
    columns=(n+scale_n-1)//scale_n
    if broadcast:
        # Independent semantic oracle only. Native admission must prove its
        # own indexing/ABI; this does not opt the compiled route into support.
        if sa.ndim < 2 or sb.ndim < 2 or sa.shape[-2:] != (m,groups) or sb.shape[-2:] != (groups,columns):
            raise ValueError("typed scaled reference matrix/scale extents differ")
        try:
            batch_shape=np.broadcast_shapes(aa.shape[:-2],bb.shape[:-2],sa.shape[:-2],sb.shape[:-2])
        except ValueError as exc:
            raise ValueError("typed scaled reference batch prefixes do not broadcast") from exc
        if min(*batch_shape,m,n,k)<=0 or kb!=k:
            raise ValueError("typed scaled reference matrix/scale extents differ")
    elif min(*batch_shape,m,n,k)<=0 or kb!=k or sa.shape!=(*lhs_batch_shape,m,groups) or sb.shape!=(*rhs_batch_shape,groups,columns) or (
            lhs_batched and rhs_batched and lhs_batch_shape!=rhs_batch_shape):
        raise ValueError("typed scaled reference matrix/scale extents differ")
    if encoded:
        # E8M0 code 0 is 2**-127; code 255 is NaN, not 2**128.
        # Decode in f64 to keep this independent oracle precise before the
        # final f32 result conversion. Encoded bytes are discrete storage,
        # never an implicit differentiable scale operand.
        sa=np.where(sa==255,np.nan,np.exp2(sa.astype(np.float64)-127))
        sb=np.where(sb==255,np.nan,np.exp2(sb.astype(np.float64)-127))
    aa=aa.astype(np.float64);bb=bb.astype(np.float64)
    out=np.zeros((*batch_shape,m,n),dtype=np.float64)
    # Reference-only arithmetic; the compiled route uses native WMMA products.
    col_blocks=np.arange(n)//scale_n
    for group in range(groups):
        lo,hi=group*scale_k,min((group+1)*scale_k,k)
        out+=(aa[...,lo:hi]@bb[...,lo:hi,:])*sa[...,group,None]*sb[...,group,col_blocks][...,None,:]
    return out.astype(np.float32)
