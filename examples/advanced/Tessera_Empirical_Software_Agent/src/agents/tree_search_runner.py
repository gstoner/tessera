import argparse, json, os, pathlib, random, shutil, time
from typing import List
from .llm_interface import LLMInterface, Proposal
from .patch_strategies import apply_patch_plan
from .scoring_api import ScorableTask, ScoreResult
from .sandbox_executor import run_candidate
from .literature_integration import load_idea_bank

class DummyLLM(LLMInterface):
    def propose(self, context: str, k: int = 4) -> List[Proposal]:
        # Placeholder proposals — replace with real model calls.
        props = []
        for i in range(k):
            props.append(Proposal(
                title=f"Try variant {i}",
                patch_plan="Replace algorithm with a slightly different setting",
                files={"main.py": f"print('hello from variant {i}')"},
                metadata={"temp": 0.8}
            ))
        return props

    def critique(self, code: str, logs: str) -> str:
        return "Looks fine. Next: try a different seed."

def simple_eval(workspace: str) -> ScoreResult:
    # Toy metric: reward programs that run successfully and print something non-empty.
    logs = run_candidate(workspace, "python main.py")
    val = 1.0 if logs.get("ok") and logs.get("stdout").strip() else 0.0
    return ScoreResult(value=val, metric="toy_success", metadata=logs)

def make_task(name: str) -> ScorableTask:
    # Swap this factory per example; keep it minimal for the skeleton.
    return ScorableTask(name=name, evaluator=simple_eval)

def run_search(task_dir: str, budget: int, k: int, out_dir: str):
    task = make_task(pathlib.Path(task_dir).name)
    llm = DummyLLM()
    os.makedirs(out_dir, exist_ok=True)

    best = (-1.0, None)
    for step in range(budget):
        ws = pathlib.Path(out_dir) / f"cand_{step:04d}"
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "main.py").write_text("print('seed')")
        proposals = llm.propose("context", k=k)
        for j, p in enumerate(proposals):
            wsj = ws / f"v{j}"
            shutil.copytree(ws, wsj, dirs_exist_ok=True)
            apply_patch_plan(str(wsj), p.patch_plan, p.files)
            score = task.score(str(wsj))
            with open(wsj / "score.json", "w") as f:
                json.dump(dict(score=score.value, metric=score.metric, meta=score.metadata), f, indent=2)
            if score.value > best[0]:
                best = (score.value, str(wsj))
        if best[0] == 1.0:
            break
    summary = dict(best_score=best[0], best_path=best[1])
    (pathlib.Path(out_dir) / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="Path to task dir")
    ap.add_argument("--budget", type=int, default=32)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--out", default="runs/ts_run")
    args = ap.parse_args()
    res = run_search(args.task, args.budget, args.k, args.out)
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
