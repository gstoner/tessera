"""Reconstruct every original disassembly hash without retaining duplicates."""
from pathlib import Path
import hashlib
import json
import lzma


def main():
    root=Path(__file__).resolve().parent
    index=json.loads((root/"isa-index.json").read_text())
    if hashlib.sha256((root/"quality-formats.json").read_bytes()).hexdigest()!=index["packet_sha256"]:
        raise RuntimeError("benchmark packet identity differs")
    for entry in index["entries"]:
        text=lzma.decompress((root/entry["canonical_file"]).read_bytes()).decode()
        if hashlib.sha256(text.encode()).hexdigest()!=entry["canonical_sha256"]:
            raise RuntimeError("canonical ISA identity differs")
        original=text.replace(entry["canonical_header"],entry["original_header"],1)
        if hashlib.sha256(original.encode()).hexdigest()!=entry["original_sha256"]:
            raise RuntimeError("original ISA reconstruction differs")
    print(f"verified {len(index['entries'])} original ISA artifacts")


if __name__=="__main__":main()
