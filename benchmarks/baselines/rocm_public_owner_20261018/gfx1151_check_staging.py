import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
parser.add_argument("--provider", type=Path)
parser.add_argument("--source-root", type=Path)
args = parser.parse_args()
p = args.root.resolve()
source_root = (args.source_root or p / "source").resolve()
info = subprocess.check_output(["/opt/rocm/bin/rocminfo"], text=True, stderr=subprocess.STDOUT)
match = re.search(r"Name:\s+gfx1151\s+Uuid:\s+(\S+)\s+Marketing Name:\s+([^\n]+)", info)
assert match, "owning gfx1151 unavailable"


class Buffer(C.Structure):
    _fields_ = [
        ("bytes", C.c_uint64),
        ("elements", C.c_uint64),
        ("first_write", C.c_int64),
        ("last_read", C.c_int64),
        ("ownership", C.c_uint32),
        ("reserved", C.c_uint32),
    ]


class Step(C.Structure):
    _fields_ = [
        ("image", C.c_void_p),
        ("image_bytes", C.c_uint64),
        ("entry", C.c_char_p),
        ("input_count", C.c_uint32),
        ("scalar_count", C.c_uint32),
        ("inputs", C.c_uint32 * 6),
        ("output", C.c_uint32),
        ("geometry", C.c_uint32 * 6),
        ("scalars", C.c_int64 * 8),
    ]


provider = (args.provider or p / "provider/libtessera_rocm_native_movement.so").resolve()
lib = C.CDLL(str(provider))
lib.tessera_rocm_program_prepare.argtypes = [
    C.c_char_p,
    C.c_uint32,
    C.c_uint32,
    C.POINTER(Buffer),
    C.c_uint32,
    C.POINTER(Step),
    C.POINTER(C.c_void_p),
    C.POINTER(C.c_uint64),
    C.POINTER(C.c_uint64),
]
lib.tessera_rocm_program_invoke.argtypes = [C.c_uint64, C.c_uint32, C.POINTER(C.c_uint64), C.POINTER(C.c_float)]
lib.tessera_rocm_program_read.argtypes = [C.c_uint64, C.c_uint32, C.c_uint64, C.c_void_p, C.c_uint64]
lib.tessera_rocm_program_update.argtypes = [C.c_uint64, C.POINTER(C.c_void_p), C.POINTER(C.c_uint64)]
lib.tessera_rocm_program_update_slots.argtypes = [
    C.c_uint64,
    C.c_uint32,
    C.POINTER(C.c_uint32),
    C.POINTER(C.c_void_p),
    C.POINTER(C.c_uint64),
]
lib.tessera_rocm_program_close.argtypes = [C.c_uint64]
lib.tessera_rocm_program_cache_stats.argtypes = [C.POINTER(C.c_uint64)] * 4
lib.tessera_rocm_program_cache_clear.argtypes = []
package = json.loads((p / "package/package.json").read_text())
raw = (p / "package/add.hsaco").read_bytes()
assert raw.startswith(b"\x7fELF")
assert package["image"]["architecture"] == "gfx1151"
image = C.create_string_buffer(raw)
desc = package["descriptor"]
geometry = desc["geometry"]
step = Step()
step.image = C.addressof(image)
step.image_bytes = len(raw)
step.entry = desc["entry_symbol"].encode()
step.input_count = 2
step.scalar_count = 1
step.inputs[:2] = [0, 1]
step.output = 2
step.geometry[:] = list(geometry["grid"]) + list(geometry["workgroup"])
n = 51
step.scalars[0] = n
contract = (Buffer * 3)(Buffer(n * 4, n, -1, 0, 0, 0), Buffer(n * 4, n, -1, 0, 0, 0), Buffer(n * 4, n, 0, 1, 2, 0))
a = (C.c_float * n)()
b = (C.c_float * n)()
inputs = (C.c_void_p * 2)(C.addressof(a), C.addressof(b))
sizes = (C.c_uint64 * 2)(n * 4, n * 4)
os.environ["TESSERA_ROCM_PROGRAM_CACHE"] = "1"
assert lib.tessera_rocm_program_cache_clear() == 0
counts = [C.c_uint64() for _ in range(4)]


def stats():
    assert lib.tessera_rocm_program_cache_stats(*[C.byref(v) for v in counts]) == 0
    return [v.value for v in counts]


before = stats()
rows = []
retained = []
for lease in range(3):
    for i in range(n):
        a[i] = lease * 100 + i
        b[i] = 2 * i - lease * 3
    expected = [float(a[i] + b[i]) for i in range(n)]
    handle = C.c_uint64()
    assert (
        lib.tessera_rocm_program_prepare(b"gfx1151", 2, 3, contract, 1, C.byref(step), inputs, sizes, C.byref(handle))
        == 0
    )
    output = (C.c_float * n)()
    generation = C.c_uint64()
    assert lib.tessera_rocm_program_read(handle, 2, 0, output, n * 4) == 10
    # Preparation completed its owned snapshot; caller mutation is not a write
    # to already uploaded device input storage.
    a[0] += 7
    event = C.c_float()
    assert lib.tessera_rocm_program_invoke(handle, 1, C.byref(generation), C.byref(event)) == 0
    assert lib.tessera_rocm_program_read(handle, 2, generation, output, n * 4) == 0
    assert list(output) == expected
    retained.append((output, expected))
    previous = generation.value
    assert lib.tessera_rocm_program_update(handle, inputs, sizes) == 0
    assert lib.tessera_rocm_program_read(handle, 2, previous, output, n * 4) == 10
    expected_update = [float(a[i] + b[i]) for i in range(n)]
    assert lib.tessera_rocm_program_invoke(handle, 1, C.byref(generation), C.byref(event)) == 0
    assert generation.value > previous
    updated = (C.c_float * n)()
    assert lib.tessera_rocm_program_read(handle, 2, generation, updated, n * 4) == 0
    assert list(updated) == expected_update
    previous = generation.value
    b[0] -= 5
    assert (
        lib.tessera_rocm_program_update_slots(
            handle, 1, (C.c_uint32 * 1)(1), (C.c_void_p * 1)(C.addressof(b)), (C.c_uint64 * 1)(n * 4)
        )
        == 0
    )
    assert lib.tessera_rocm_program_read(handle, 2, previous, updated, n * 4) == 10
    assert list(updated) == expected_update
    assert lib.tessera_rocm_program_invoke(handle, 1, C.byref(generation), C.byref(event)) == 0
    partial = (C.c_float * n)()
    assert lib.tessera_rocm_program_read(handle, 2, generation, partial, n * 4) == 0
    assert list(partial) == [float(a[i] + b[i]) for i in range(n)]
    assert generation.value > previous
    assert lib.tessera_rocm_program_close(handle) == 0
    assert lib.tessera_rocm_program_invoke(handle, 1, C.byref(generation), None) == 1
    row = stats()
    assert row[2] == 1 and 0 < row[3] <= 128 << 20
    rows.append(dict(lease=lease, cache=row, correctness="exact_integer_addition"))
    for old, wanted in retained:
        assert list(old) == wanted
after = stats()
assert after[0] - before[0] == 2
assert lib.tessera_rocm_program_cache_clear() == 0
assert stats()[2:] == [0, 0]
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
record = dict(
    architecture="gfx1151",
    device=match[2].strip(),
    uuid=match[1],
    recorder_sha256=sha(Path(__file__)),
    provider_sha256=sha(provider),
    image_sha256=hashlib.sha256(raw).hexdigest(),
    rows=rows,
    source_files={str(f.relative_to(source_root)): sha(f) for f in source_root.rglob("*") if f.is_file()},
    scope="native owner staging/reuse/generation correctness with compiler-generated Graph/Schedule/Tile add;not gfx1201 format proof or performance",
)
(p / "staging-proof.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record))
