
import json, os, platform, subprocess, shutil

def which(cmd):
    return shutil.which(cmd) is not None

def try_cmd(cmd):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=5)
    except Exception as e:
        return f"error: {e}"

def collect():
    env = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "cpu": platform.processor(),
        "env_vars": {k: os.environ.get(k, "") for k in ["CUDA_VISIBLE_DEVICES","HIP_VISIBLE_DEVICES"]},
        "tools": {
            "nvidia_smi": which("nvidia-smi"),
            "amd_smi": which("amd-smi")
        }
    }
    if env["tools"]["nvidia_smi"]:
        env["nvidia_smi"] = try_cmd(["nvidia-smi","-L"])
    if env["tools"]["amd_smi"]:
        env["amd_smi"] = try_cmd(["amd-smi","list","--json"])
    return env

if __name__ == "__main__":
    print(json.dumps(collect(), indent=2))
