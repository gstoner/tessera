# Integrating Gemma → Tessera Port

Place this `models/gemma_tessera/` directory at the root of your repo, then:

```bash
pip install -e models/gemma_tessera         # installs tessera_gemma package
pytest -q models/gemma_tessera/tests
```

To convert weights (after accepting Gemma terms and downloading a HF checkpoint):
```bash
python models/gemma_tessera/scripts/convert_hf_gemma_to_tessera.py --hf-id google/gemma-2-2b-it --out models/gemma_tessera/weights_tessera.pt
```
