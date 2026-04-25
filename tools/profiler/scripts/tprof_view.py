#!/usr/bin/env python3
import http.server, socketserver, webbrowser, argparse, os, sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Directory to serve (default: .)")
    ap.add_argument("--file", default="demo.report.html", help="HTML file to open (relative to root)")
    ap.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    ap.add_argument("--no-open", action="store_true", help="Do not open browser automatically")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    html_path = root / args.file
    if not html_path.exists():
        print(f"[tprof_view] File not found: {html_path}", file=sys.stderr)
        sys.exit(2)

    os.chdir(root)
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("127.0.0.1", args.port), handler) as httpd:
        url = f"http://127.0.0.1:{args.port}/{args.file}"
        print(f"[tprof_view] Serving {root} at {url}")
        if not args.no_open:
            try:
                webbrowser.open(url, new=2)
            except Exception:
                pass
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[tprof_view] Shutting down")

if __name__ == "__main__":
    main()
