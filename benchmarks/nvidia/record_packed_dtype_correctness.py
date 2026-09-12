#!/usr/bin/env python3
"""Nonzero packed/scaled decode probes with padding and both packing axes."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import ml_dtypes
from benchmarks.record_dtype_arithmetic import check


def case(logical, axis):
    rows, columns, origin, offset = 9, 17, 1, 3
    factor = 1 if logical.startswith('fp6') else 2
    physical_rows, physical_columns = rows + origin, columns + origin
    cr = (physical_rows + factor-1)//factor if axis == 0 else physical_rows
    cc = (physical_columns + factor-1)//factor if axis == 1 else physical_columns
    stride = cc + 3
    source = np.full(offset + cr*stride + 3, 0xC0 if factor == 1 else 0, np.uint8)
    codes = np.arange(physical_rows*physical_columns, dtype=np.uint8).reshape(physical_rows, physical_columns)
    codes %= 64 if factor == 1 else 16
    for r in range(physical_rows):
        for c in range(physical_columns):
            rr, col, lane = (r//factor,c,r%factor) if axis == 0 else (r,c//factor,c%factor)
            source[offset + rr*stride+col] |= int(codes[r,c]) << (4*lane)
    if logical == 'int4':
        expected = np.where(codes < 8,codes.astype(np.int32),codes.astype(np.int32)-16).astype(np.float32)
    else:
        dtype = {'nvfp4':ml_dtypes.float4_e2m1fn, 'fp4_e2m1':ml_dtypes.float4_e2m1fn,
                 'fp6_e2m3':ml_dtypes.float6_e2m3fn, 'fp6_e3m2':ml_dtypes.float6_e3m2fn}[logical]
        expected = codes.view(dtype).astype(np.float32)
    scaled = logical != 'int4'
    scale_stride = (physical_columns if axis == 0 else (physical_columns+3)//4) + 2
    scale = np.full(2 + physical_rows*scale_stride,127,np.uint8)
    for r in range(physical_rows):
        for c in range(physical_columns):
            sr, sc = (r//4, c) if axis == 0 else (r, c//4)
            exponent = (sr+sc)%3-1
            scale[2+sr*scale_stride+sc] = (7+exponent)*8 if logical == 'nvfp4' else 127+exponent
            if scaled:
                expected[r,c] *= 2.**exponent
    kwargs = dict(logical=logical,rows=rows,columns=columns,source_bytes=source.size,
        packing_axis=axis,strides=(stride,1),offset=offset)
    if scaled:
        kwargs.update(scale_dtype='ue4m3' if logical=='nvfp4' else 'ue8m0',scale_bytes=scale.size,
                      scale_block_size=4,scale_axis=axis,scale_layout='row_major',scale_stride=scale_stride,scale_offset=2)
    if not scaled:
        scale = np.zeros(1, np.uint8)
    output = np.full((rows,columns),np.nan,np.float32)
    args = dict(source=source,scale=scale,output=output,RowOrigin=origin,ColumnOrigin=origin,
                Rows=rows,Columns=columns,SourceBytes=source.size,ScaleBytes=scale.size)
    return kwargs,args,expected[origin:,origin:].copy()


def record():
    from tessera.compiler.nvidia_native import package_packed_decode
    from tessera.runtime import RuntimeArtifact, launch
    rows=[]
    for logical in ('int4','nvfp4','fp4_e2m1','fp6_e2m3','fp6_e3m2'):
        for axis in (0,1):
            kwargs,args,expected=case(logical,axis)
            package=package_packed_decode(**kwargs)
            artifact=RuntimeArtifact(tile_ir=package.tile_ir,target_ir=package.target_ir,
                metadata={'target':'nvidia_sm120'},native_image=package.image,launch_descriptor=package.descriptor)
            receipt=launch(artifact,args)
            if not receipt.get('ok') or receipt.get('execution_kind') not in ('native_gpu', 'native'):
                raise RuntimeError(f'no native packed execution receipt: {receipt}')
            errors=check(args['output'],expected)
            rows.append(dict(logical=logical,packing_axis=axis,mismatches=errors,
                state='passed' if errors==0 else 'numerical_failure',promotion_eligible=False,
                image_sha256=hashlib.sha256(package.image.payload).hexdigest(),
                compiler_fingerprint=package.image.compiler_fingerprint,
                toolchain_fingerprint=package.image.toolchain_fingerprint,
                tile_sha256=hashlib.sha256(package.tile_ir.encode()).hexdigest(),
                source_sha256=hashlib.sha256(args['source'].tobytes()).hexdigest(),
                scale_sha256=hashlib.sha256(args['scale'].tobytes()).hexdigest()))
    return dict(backend='nvidia',scope='generic packed-load numerical/layout proof, not matrix acceleration',rows=rows)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    result=record()
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    if any(r['state']!='passed' for r in result['rows']):
        raise SystemExit(1)
