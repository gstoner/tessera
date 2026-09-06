#!/usr/bin/env python3
"""Owning-Mac proof of serialized native arena packages on resident tensors."""
import argparse
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
import numpy as np
from tessera.compiler.apple_native_arena import AppleNativeArena, AppleArenaPackage, build_apple_arena_package
from tessera.runtime import DeviceTensor


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--typed', action='store_true')
    args = p.parse_args()
    data = json.loads(args.artifact.read_text())
    digest = data.pop('digest')
    artifact = AppleNativeArena(**data)
    if artifact.digest != digest:
        raise ValueError('native artifact identity disagrees')
    package = build_apple_arena_package(artifact)
    package = AppleArenaPackage.from_json(package.to_json(), expected_digest=package.binding_digest)
    dispatch = None
    if args.typed:
        import tessera as ts
        @ts.jit
        def dispatch(output, n, rounds):
            return output
        dispatch.bind_apple_native_arena(package)
        assert dispatch.execution_kind == 'native_gpu'
        assert dispatch.runtime_artifact().metadata['package_digest'] == package.binding_digest
    rows = []
    if not DeviceTensor.is_metal():
        raise RuntimeError('owning Metal device unavailable')
    with package.bind() as bound:
        for width, rounds in [(1, 0), (17, 1), (32, 7), (64, 17), (128, 17), (256, 33)]:
            output = DeviceTensor.empty((32, width), np.float32)
            if output is None:
                raise RuntimeError('resident output allocation failed')
            try:
                if dispatch is not None:
                    assert dispatch(output, width, rounds) is output
                    size = bound._size(None, width, rounds)
                    # Keyword dispatch and lazy reopening use the same manifest.
                    dispatch.close_native_storage()
                    assert dispatch(output=output, n=width, rounds=rounds) is output
                    wrong_shape = DeviceTensor.empty((32, width + 1), np.float32)
                    try:
                        try:
                            dispatch(wrong_shape, width, rounds)
                        except ValueError as exc:
                            assert 'shape/dtype' in str(exc)
                        else:
                            raise AssertionError('typed tensor shape contract was ignored')
                    finally:
                        wrong_shape.free()
                    for bad in (True, 0, 257):
                        try:
                            dispatch(output, bad, rounds)
                        except ValueError:
                            pass
                        else:
                            raise AssertionError('typed scalar contract was ignored')
                else:
                    size = bound.launch((output, width, rounds), grid=(32, 1, 1), block=(width, 1, 1))
                expected = np.tile(rounds * ((np.arange(width) + 1) % width) + rounds * (rounds - 1) / 2, (32, 1))
                np.testing.assert_array_equal(output.copy_to_host(), expected)
                rows.append(dict(width=width, rounds=rounds, native_bytes=size, oracle='exact'))
                for invalid in [True, -1, (1 << 63) - 1]:
                    try:
                        bound.launch((output, invalid, rounds), grid=(32, 1, 1), block=(width, 1, 1))
                    except (ValueError, RuntimeError):
                        pass
                    else:
                        raise AssertionError('invalid sizing input was admitted')
                view = output.reshape_view((32, width))
                try:
                    bound.launch((view, width, rounds), grid=(32, 1, 1), block=(width, 1, 1))
                except ValueError:
                    pass
                else:
                    raise AssertionError('non-owning view was admitted')
                original_shape = output.shape
                output.shape = (output.nbytes + 4096,)
                try:
                    bound.launch((output, width, rounds), grid=(32, 1, 1), block=(width, 1, 1))
                except RuntimeError as exc:
                    assert 'extent/device' in str(exc)
                else:
                    raise AssertionError('forged resident extent was admitted')
                finally:
                    output.shape = original_shape
            finally:
                output.free()
    if dispatch is not None:
        dispatch.close_native_storage()
    runtime = Path(os.environ['TESSERA_APPLE_GPU_RUNTIME_LIB'])
    args.output.write_text(json.dumps(dict(typed_jit=args.typed, rows=rows, package_digest=package.binding_digest,
        device=subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
        platform=platform.platform(), runtime_sha256=hashlib.sha256(runtime.read_bytes()).hexdigest(),
        binding_sha256=hashlib.sha256((Path(__file__).resolve().parents[1] / 'python/tessera/compiler/apple_native_arena.py').read_bytes()).hexdigest(),
        compiler_sha256=artifact.compiler_digest,
        jit_sha256=hashlib.sha256((Path(__file__).resolve().parents[1] / 'python/tessera/compiler/jit.py').read_bytes()).hexdigest(),
        clang=subprocess.check_output(['xcrun', 'clang', '--version'], text=True),
        artifact_digest=artifact.digest, bridge_sha256=package.bridge_digest,
        recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        placement='resident Metal DeviceTensor on runtime device and queue', selector_promotion=False,
        boundary='explicit raw buffer/index package; caller owns extents and launch geometry'), indent=2) + '\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
