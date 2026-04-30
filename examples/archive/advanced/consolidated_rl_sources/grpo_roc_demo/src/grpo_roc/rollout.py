
import re, ast
from dataclasses import dataclass
from typing import Dict, Any

TOOL_CALL_RE = re.compile(r'tool_call\{(?P<json>.*?)\}/tool_call', re.S)
ANSWER_TAG_RE = re.compile(r'answeranswer(.*?)\/answeranswer', re.S)
BOXED_RE = re.compile(r'\\boxed\{\s*([+-]?\d+)\s*\}')

@dataclass
class ToolExec:
    ok: bool
    out: str

def _safe_eval_numeric(src: str, timeout_s: float=1.0) -> ToolExec:
    try:
        node = ast.parse(src, mode='eval')
        allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.FloorDiv, ast.Mod, ast.USub, ast.UAdd, ast.Load, ast.Constant)
        if not all(isinstance(n, allowed) for n in ast.walk(node)):
            return ToolExec(False, "disallowed AST node")
        for n in ast.walk(node):
            if isinstance(n, ast.Name):
                return ToolExec(False, "names disabled in demo")
            if isinstance(n, ast.Call):
                return ToolExec(False, "calls disabled in demo")
        val = eval(compile(node, '<sandbox>', 'eval'), {'__builtins__': {}}, {})
        return ToolExec(True, str(val))
    except Exception as e:
        return ToolExec(False, f"error: {e}")

def run_one_turn(text: str, tool_timeout_s: float=1.0) -> (str, Dict[str, Any]):
    info = {"tool_calls": 0, "tool_errors": 0}
    out = text
    for m in TOOL_CALL_RE.finditer(text):
        info["tool_calls"] += 1
        try:
            js = m.group('json')
            code_m = re.search(r'\"code\"\s*:\s*\"(.*?)\"', js, re.S)
            code = code_m.group(1).encode('utf-8').decode('unicode_escape') if code_m else ""
            exec_res = _safe_eval_numeric(code, timeout_s=tool_timeout_s)
            if exec_res.ok:
                resp = f"tool_response{{\"stdout\": \"{exec_res.out}\"}}/tool_response"
            else:
                resp = f"tool_response{{\"error\": \"{exec_res.out}\"}}/tool_response"
                info["tool_errors"] += 1
            out += "\n" + resp + "\n"
        except Exception as e:
            info["tool_errors"] += 1
            out += f"\ntool_response{{\"error\": \"exception: {e}\"}}/tool_response\n"
    return out, info

def extract_answer_and_format(text: str) -> (int, Dict[str,int]):
    tags = ANSWER_TAG_RE.findall(text)
    fmt = {"answer_tags": len(tags)}
    ans = None
    if tags:
        boxed = BOXED_RE.findall(tags[-1])
        if boxed:
            ans = int(boxed[-1])
    return ans, fmt
