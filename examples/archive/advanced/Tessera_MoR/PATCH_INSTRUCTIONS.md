# PATCH_INSTRUCTIONS.md
This bundle is organized with **repo-relative paths** so you can drop it into the root of `gstoner/tessera`.

## Steps
1. Unzip at repo root (you should see `examples/mor_token_choice/` and `.github/workflows/mor.yml`).
2. (Optional) In your top-level `CMakeLists.txt`, add:
   ```cmake
   add_subdirectory(examples/mor_token_choice)
   ```
3. Commit:
   ```bash
   git add examples/mor_token_choice .github/workflows/mor.yml
   git commit -m "examples: add MoR token-choice Tessera test app with lit + CI"
   ```

Generated: 2025-09-03T21:11:37.021697Z
