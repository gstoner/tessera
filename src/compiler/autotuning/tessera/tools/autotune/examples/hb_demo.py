#!/usr/bin/env python3
import os, sys
from tessera.tools.autotune.tessera_autotuner.cli import main
if __name__ == "__main__":
    sys.argv = [
        "autotune",
        "--config", os.path.join(os.path.dirname(__file__), "..","configs","matmul_sm90.json"),
        "--algo", "hyperband",
        "--hb-min-budget", "5",
        "--hb-max-budget", "30",
        "--hb-eta", "3",
        "-o", os.path.join(os.path.dirname(__file__), "..","runs","hb_demo")
    ]
    main()
