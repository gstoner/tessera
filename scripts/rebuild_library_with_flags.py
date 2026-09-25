#!/usr/bin/env python3
"""Rebuild one shared library from a build tree's own ninja commands with extra
flags, into a separate directory. The build tree is never modified.

Usage: rebuild_library_with_flags.py <tree> <lib relpath> <outdir> [flag ...]

Only the objects on the library's own link line are recompiled. With no extra
flags the result should be byte-identical to the tree's library -- check that
first (``cmp``); it is what makes an optimized copy a controlled comparison.
Load the copy through the runtime's path override (TESSERA_X86_ELEMENTWISE_LIB,
TESSERA_ROCM_SPECTRAL_LIB, TESSERA_NVIDIA_FFT_LIB, TESSERA_APPLE_GPU_RUNTIME_LIB).
Evidence: benchmarks/baselines/runtime_lib_opt_20260925/README.md."""
import os, re, shlex, subprocess, sys
tree, target, outdir, extra = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
os.makedirs(outdir, exist_ok=True)
cmds = subprocess.run(["ninja", "-C", tree, "-t", "commands", target], capture_output=True,
                      text=True, check=True).stdout.splitlines()
link = next(c for c in cmds if re.search(r"-o\s+" + re.escape(target) + r"(\s|$)", c))
segment = next(part for part in link.split("&&") if re.search(r"-o\s+" + re.escape(target), part))
link = segment.strip()
objs = [t for t in shlex.split(link) if t.endswith(".o")]
mapping = {}
for c in cmds:
    m = re.search(r"\s-o\s+(\S+\.o)\s", c + " ")
    if m and m.group(1) in objs:
        new = os.path.join(outdir, m.group(1).replace("/", "_"))
        mapping[m.group(1)] = new
        c = c.replace(" -o " + m.group(1), " -o " + new, 1)
        c = re.sub(r"-MD -MT \S+ -MF \S+", "", c)
        c = re.sub(r"--?dependency-file=\S+|-MD\s+-MF\s+\S+", "", c)
        rc = subprocess.run(c + " " + " ".join(extra), shell=True, cwd=tree, capture_output=True, text=True)
        if rc.returncode:
            sys.exit(f"compile failed: {c[:200]}\n{rc.stderr[-800:]}")
missing = [o for o in objs if o not in mapping]
if missing:
    sys.exit(f"no compile command for {missing[:3]}")
out = os.path.join(outdir, os.path.basename(target))
for o, n in mapping.items():
    link = link.replace(o, n)
link = re.sub(r"-o\s+" + re.escape(target) + r"(\s|$)", "-o " + out + r"\1", link)
link = re.sub(r"-Wl,--dependency-file=\S+", "", link)
rc = subprocess.run(link, shell=True, cwd=tree, capture_output=True, text=True)
if rc.returncode:
    sys.exit(f"link failed:\n{rc.stderr[-800:]}")
if not mapping:
    sys.exit("no objects rebuilt")
print(f"built {out} from {len(mapping)} objects with {' '.join(extra)}")
