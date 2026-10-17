from pathlib import Path
import json
import shlex
import subprocess

r = Path("/home/angstorms/scratch/sm120-relu-producer-20261014")
w = Path("/home/angstorms/scratch/rocm-native-gate-up-views-20261016")
commands = json.loads((r / ".build-relu/compile_commands.json").read_text())
entry = next(c for c in commands if c["file"].endswith("/tools/tessera-opt/tessera-opt.cpp"))
args = shlex.split(entry["command"])
out = args.index("-o")
args[out + 1] = str(w / "slice-contract-normalize.o")
args[-1] = str(w / "slice-contract-normalize.cpp")
subprocess.run(args, cwd=entry["directory"], check=True)
link = subprocess.check_output(
    ["ninja", "-C", str(r / ".build-relu"), "-t", "commands", "tessera-opt"], text=True
).splitlines()[-1]
args = shlex.split(link.split("&&", 1)[1].rsplit("&&", 1)[0].strip())
args = [a for a in args if not a.startswith("-Wl,--dependency-file=")]
args = [str(w / "slice-contract-normalize.o") if a.endswith("tessera-opt.cpp.o") else a for a in args]
out = args.index("-o")
args[out + 1] = str(w / "slice-contract-normalize")
(w / "buffer-view-link-command.json").write_text(json.dumps(args, indent=2) + "\n")
subprocess.run(args, cwd=r / ".build-relu", check=True)
print("Built isolated actual Graph slice contract harness; production compiler unchanged")
