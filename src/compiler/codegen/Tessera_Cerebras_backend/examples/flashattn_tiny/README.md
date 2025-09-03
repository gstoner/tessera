# FlashAttention (tiny) example (scaffold)

- `flashattn_tiny.csl` — CSL-like kernel scaffold.
- `host_run_flashattn.py` — host stub that shows inputs/outputs and a `scale` scalar.

**Build & Run (conceptual):**
```bash
cs_compiler flashattn_tiny.csl -o flashattn_tiny.elf
python host_run_flashattn.py --elf ./flashattn_tiny.elf
```
