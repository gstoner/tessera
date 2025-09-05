
import subprocess, shutil, json

def _which(x): return shutil.which(x) is not None

def _try(cmd, timeout=5):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=timeout)
    except Exception as e:
        return f"error: {e}"

def probe_gpu_links():
    out = {"nvidia_smi": None, "nvlink_topo": None, "amd_smi": None, "lspci_gpu": None}
    if _which("nvidia-smi"):
        out["nvidia_smi"] = _try(["nvidia-smi", "-L"])
        topo = _try(["nvidia-smi", "topo", "-m"])
        out["nvlink_topo"] = topo
    if _which("amd-smi"):
        out["amd_smi"] = _try(["amd-smi", "list", "--json"])
    if _which("lspci"):
        out["lspci_gpu"] = _try(["bash","-lc","lspci | egrep -i 'vga|3d|display'"])
    return out

def probe_nic():
    out = {"ibv_devinfo": None, "ibstat": None, "ethtool": None}
    if _which("ibv_devinfo"):
        out["ibv_devinfo"] = _try(["ibv_devinfo"])
    if _which("ibstat"):
        out["ibstat"] = _try(["ibstat"])
    if _which("bash"):
        out["ethtool"] = _try(["bash","-lc","for d in /sys/class/net/*; do n=$(basename $d); ethtool $n 2>/dev/null | head -n 20; done"])
    return out

def collect_all():
    return {
        "gpu_links": probe_gpu_links(),
        "nic": probe_nic()
    }

if __name__ == "__main__":
    print(json.dumps(collect_all(), indent=2))
