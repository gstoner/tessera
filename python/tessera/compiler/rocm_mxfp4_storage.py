"""Lossless reference contract for the native gfx1201 MXFP4 storage bridge."""
import numpy as np

MXFP4_STORAGE_CONTRACT = "mxfp4.gfx12.n16_k16_lane_u32.plus_row_reference.v1"

def checked_storage_inputs(codes,exponents,*,storage_contract):
    if storage_contract!=MXFP4_STORAGE_CONTRACT:
        raise ValueError("MXFP4 storage bridge requires its exact lossless storage contract")
    codes,exponents=np.asarray(codes),np.asarray(exponents)
    if codes.dtype!=np.uint8 or codes.ndim!=2:
        raise ValueError("MXFP4 storage bridge requires uint8 packed [N,K/2]")
    n,half_k=codes.shape
    k=half_k*2
    if n<=0 or n%16 or k<=0 or k%64:
        raise ValueError("MXFP4 storage bridge requires positive N16/K64")
    if exponents.dtype!=np.uint8 or exponents.shape!=(k//32,n):
        raise ValueError("MXFP4 storage bridge requires uint8 group-major [K32,N]")
    return codes,exponents

def reference_mxfp4_folded_storage(codes,exponents,*,storage_contract):
    """Byte permutation and unsigned maximum only; no quantization or folding."""
    from .rocm_mxfp4 import convert_weight_layout,MXFP4_CHECKPOINT_LAYOUT_V1,MXFP4_GFX12_FRAGMENT_LAYOUT_V1
    codes,exponents=checked_storage_inputs(codes,exponents,storage_contract=storage_contract)
    fragment=convert_weight_layout(codes,source=MXFP4_CHECKPOINT_LAYOUT_V1,
        destination=MXFP4_GFX12_FRAGMENT_LAYOUT_V1)
    plane=np.concatenate((exponents,exponents.max(axis=0,keepdims=True)),axis=0)
    return np.ascontiguousarray(fragment),np.ascontiguousarray(plane)
