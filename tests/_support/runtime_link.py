"""What an out-of-CMake consumer must add to link `libtessera_runtime.a`.

The archive carries the CUDA/HIP backend objects whenever the build enabled
them, and those objects need `libcudart` / `libamdhip64` at link time. Inside the
build, CMake supplies that through the target's usage requirements; a harness
compiled with its own command line gets nothing, and the link fails with a page
of `undefined reference to hipMalloc` that reads as a broken ABI rather than a
missing library.

`src/runtime/CMakeLists.txt` writes the answer beside the archive
(`tessera_runtime.consumer-link.txt`, one absolute path per line, empty when no
device backend is compiled in). This reads it. A build tree from before that
sidecar existed returns `()`, which is exactly the old behaviour — correct on a
CPU-only build and still a link error on a device build, so an old tree degrades
to the status quo rather than to a wrong answer.
"""
from __future__ import annotations

from pathlib import Path


def runtime_consumer_link_args(runtime_lib: Path) -> tuple[str, ...]:
    """Extra link arguments for a harness linking ``runtime_lib`` directly."""
    sidecar = Path(runtime_lib).parent / "tessera_runtime.consumer-link.txt"
    try:
        text = sidecar.read_text(encoding="utf-8")
    except OSError:
        return ()
    return tuple(line.strip() for line in text.splitlines() if line.strip())
