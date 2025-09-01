import argparse, os, re
from pathlib import Path

def find_label_htmls(root: Path):
    items = []
    for child in root.iterdir():
        # Expect artifacts named runs-<label>/ with nested runs/flashattn_sweep_report.html
        if not child.is_dir(): continue
        label = child.name.replace("runs-","")
        html = child / "runs" / "flashattn_sweep_report.html"
        if html.exists():
            items.append((label, html))
    return items

def rewrite_paths(html_text: str, prefix: str):
    # replace src='runs/...' with src='<prefix>/runs/...'
    html_text = re.sub(r"src=['"](runs/[^'"]+)['"]", lambda m: f"src='{prefix}/{m.group(1)}'", html_text)
    return html_text

def make_tabs(pieces):
    # Simple CSS tabs
    head = """<html><head><meta charset='utf-8'><title>Tessera Weekly Combined Report</title>
<style>
.tabs{display:flex;cursor:pointer;margin-bottom:10px;}
.tab{padding:8px 12px;background:#eee;margin-right:6px;border-radius:6px;}
.tab.active{background:#ccc;}
.panel{display:none;} .panel.active{display:block;}
</style>
</head><body>
<h1>Tessera Weekly Combined Report</h1>
<div class='tabs'>"""
    body = []
    for i,(label,_) in enumerate(pieces):
        cls = "tab active" if i==0 else "tab"
        head += f"<div class='{cls}' data-idx='{i}'>{label}</div>"
    head += "</div>"
    for i,(label,content) in enumerate(pieces):
        pcls = "panel active" if i==0 else "panel"
        body.append(f"<div class='{pcls}' id='panel-{i}'>{content}</div>")
    tail = """<script>
document.querySelectorAll('.tab').forEach(t=>t.addEventListener('click',()=>{
  const idx = t.getAttribute('data-idx');
  document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(x=>x.classList.remove('active'));
  t.classList.add('active');
  document.getElementById('panel-'+idx).classList.add('active');
}));
</script>
</body></html>"""
    return head + "\n".join(body) + tail

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    root = Path(args.root)
    items = find_label_htmls(root)
    pieces = []
    for label, html_path in items:
        txt = html_path.read_text(encoding='utf-8')
        # prefix is artifact directory name
        prefix = html_path.parent.parent.name
        txt = rewrite_paths(txt, prefix)
        # strip outer <html>...</html> if present
        txt = re.sub(r"(?is)^.*?<body>(.*)</body>.*$", r"\1", txt)
        pieces.append((label, txt))
    combined = make_tabs(pieces)
    Path(args.out).write_text(combined, encoding='utf-8')
    print("Wrote", args.out)

if __name__ == "__main__":
    main()
