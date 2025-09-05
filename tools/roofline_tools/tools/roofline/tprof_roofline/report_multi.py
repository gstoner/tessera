from typing import List, Tuple
import base64
from .model import RooflineResult, CommEvent
from .plot import plot_roofline_with_comm

HTML_TMPL = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8"/>
<title>Roofline Multi-Device</title>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 20px; }
.tab { overflow: hidden; border-bottom: 1px solid #ccc; }
.tab button { background: #eee; border: none; outline: none; padding: 10px 14px; cursor: pointer; }
.tab button.active { background: #666; color: #fff; }
.tabcontent { display: none; padding: 12px 0; }
.figure { text-align: center; margin: 1rem auto; }
</style>
<script>
function openTab(evt, name) {
  var i, tabcontent, tablinks;
  tabcontent = document.getElementsByClassName("tabcontent");
  for (i = 0; i < tabcontent.length; i++) { tabcontent[i].style.display = "none"; }
  tablinks = document.getElementsByClassName("tablink");
  for (i = 0; i < tablinks.length; i++) { tablinks[i].className = tablinks[i].className.replace(" active", ""); }
  document.getElementById(name).style.display = "block";
  evt.currentTarget.className += " active";
}
window.onload = function(){ document.getElementsByClassName('tablink')[0].click(); }
</script>
</head><body>
<h1>Roofline — Multi-Device</h1>
<div class="tab">
{tab_buttons}
</div>
{tab_panels}
</body></html>
"""

def _b64_png(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")

def generate_multi(dev_results: List[Tuple[str, str]], out_html: str="roofline_multi.html") -> str:
    # dev_results: list of (device_name, fig_path)
    btns = []
    panels = []
    for i, (name, fig) in enumerate(dev_results):
        btns.append(f"<button class=\"tablink\" onclick=\"openTab(event, 'tab{i}')\">{name}</button>")
        panels.append(f"<div id=\"tab{i}\" class=\"tabcontent\"><div class=\"figure\"><img src=\"data:image/png;base64,{_b64_png(fig)}\"/></div></div>")
    html = HTML_TMPL.format(tab_buttons="\n".join(btns), tab_panels="\n".join(panels))
    with open(out_html, "w") as f:
        f.write(html)
    return out_html
