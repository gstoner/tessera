"""Very small sandbox runner stub — replace with your hardened runner."""
import subprocess, json, os, time, pathlib, tempfile

def run_candidate(workspace: str, cmd: str, timeout_s: int = 600):
    start = time.time()
    try:
        out = subprocess.run(cmd, cwd=workspace, shell=True, timeout=timeout_s,
                             capture_output=True, text=True)
        rt = time.time() - start
        return dict(
            ok=(out.returncode == 0),
            stdout=out.stdout, stderr=out.stderr,
            returncode=out.returncode, runtime_s=rt
        )
    except subprocess.TimeoutExpired:
        return dict(ok=False, stdout="", stderr="timeout", returncode=124, runtime_s=timeout_s)
