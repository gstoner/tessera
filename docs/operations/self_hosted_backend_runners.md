# Dedicated Hardware Runners

Each physical backend machine is a single-purpose GitHub Actions runner.  A
hardware proof must request its exact-device label, not merely `self-hosted` or
a broad backend label.  This prevents one device's evidence from being
attributed to another architecture.

| Physical host | Runner name | Exact workflow label | Supplemental labels |
|---|---|---|---|
| RTX 5070 Ti, CUDA compute capability 12.0 | `tessera-nvidia-rtx5070ti-sm120` | `nvidia-rtx5070ti-sm120` | `nvidia-sm120`, `nvidia` |
| Ryzen AI Max+ 395 / Radeon 8060S, ROCm gfx1151 | `tessera-rocm-strix-halo-gfx1151` | `rocm-strix-halo-gfx1151` | `rocm-gfx1151`, `rocm` |
| Apple M1 Max, Metal Apple7 | `tessera-apple-m1max-apple7` | `apple-m1max-apple7` | `apple7`, `apple-gpu` |

GitHub automatically supplies `self-hosted` and the operating-system label.
The NVIDIA release gate therefore uses:

```yaml
runs-on: [self-hosted, linux, nvidia-rtx5070ti-sm120]
```

## Register a runner

Run this on the matching physical host from a checkout with authenticated
`gh`.  The script verifies the attached hardware before it obtains a
short-lived registration token, and does not print or persist that token.

```bash
scripts/configure_backend_runner.sh nvidia-rtx5070ti-sm120 --install-user-service
scripts/configure_backend_runner.sh rocm-strix-halo-gfx1151 --install-user-service
scripts/configure_backend_runner.sh apple-m1max-apple7
```

For the two Linux/WSL runners, make the user service survive logout once:

```bash
sudo loginctl enable-linger "$USER"
```

For the Apple host, install the runner's supplied launchd service after
registration:

```bash
cd "$HOME/actions-runner/tessera-apple-m1max-apple7"
sudo ./svc.sh install
sudo ./svc.sh start
```

Do not place a second backend's label on any of these runners.  If hardware is
replaced, remove the old runner in GitHub, choose a new exact label, and update
the corresponding workflow and evidence documentation together.
