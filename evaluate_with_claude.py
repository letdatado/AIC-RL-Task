# -*- coding: utf-8 -*-
"""
evaluate_with_claude.py

Reads prompt.md, asks Claude to implement macro_f1, writes code to starter/solution.py,
runs the grader, and repeats N times. Saves raw/model code per trial for debugging.
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

try:
    import anthropic
except Exception as e:
    print("Please install anthropic first: uv pip install anthropic")
    sys.exit(1)

ROOT = Path(__file__).resolve().parent
PROMPT_PATH = ROOT / "prompt.md"
SOLUTION_PATH = ROOT / "starter" / "solution.py"
GRADER_PATH = ROOT / "grader" / "grade.py"
RUNS_DIR = ROOT / "runs"

CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(?P<code>[\s\S]*?)```", re.IGNORECASE)

def read_prompt() -> str:
    base = PROMPT_PATH.read_text(encoding="utf-8")
    system_header = (
        "You are an expert Python ML engineer. "
        "Return ONLY the contents of starter/solution.py implementing macro_f1(y_true, y_pred). "
        "No explanations. No extra files. Wrap your answer in a single ```python fenced block."
    )
    postscript = (
        "\n\nImplementation reminders:\n"
        "- Do not mutate inputs\n"
        "- Accept lists or numpy arrays\n"
        "- Handle zero-division as specified\n"
        "- Return a float in [0,1]\n"
        "- Do not import external libraries beyond numpy (optional)\n"
    )
    return system_header + "\n\n" + base + postscript

def extract_code(text: str) -> str:
    m = CODE_BLOCK_RE.search(text or "")
    code = m.group("code") if m else (text or "")
    return code.replace("\r\n", "\n").lstrip("\ufeff").strip()

def write_solution(code: str) -> None:
    SOLUTION_PATH.write_text(code, encoding="utf-8")

def run_grader() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(GRADER_PATH)],
        capture_output=True, text=True, cwd=str(ROOT)
    )

def call_claude(prompt: str, model: str, temperature: float, max_tokens: int) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set.")
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system="Follow the instructions precisely and return only valid Python.",
        messages=[{"role":"user","content": prompt}]
    )
    parts = [c.text for c in msg.content if getattr(c, "type", "") == "text"]
    return "\n".join(parts).strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--num_trials", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--model", type=str, default="claude-3-5-sonnet-latest")
    ap.add_argument("--max_tokens", type=int, default=1400)
    args = ap.parse_args()

    RUNS_DIR.mkdir(exist_ok=True)
    prompt = read_prompt()

    passes = 0
    for i in range(1, args.num_trials + 1):
        trial_dir = RUNS_DIR / f"trial_{i:02d}"
        trial_dir.mkdir(exist_ok=True)
        print(f"\n=== Trial {i}/{args.num_trials} ===")

        try:
            raw = call_claude(prompt, args.model, args.temperature, args.max_tokens)
        except Exception as e:
            print(f"API error: {e}")
            (trial_dir / "error.txt").write_text(str(e), encoding="utf-8")
            continue

        (trial_dir / "raw.txt").write_text(raw, encoding="utf-8")

        code = extract_code(raw)
        if "def macro_f1(" not in code:
            print("Response did not contain function signature. FAIL.")
            (trial_dir / "extracted.txt").write_text(code, encoding="utf-8")
            continue

        (trial_dir / "solution.py").write_text(code, encoding="utf-8")
        write_solution(code)

        proc = run_grader()
        (trial_dir / "grader_stdout.txt").write_text(proc.stdout, encoding="utf-8")
        if proc.stderr:
            (trial_dir / "grader_stderr.txt").write_text(proc.stderr, encoding="utf-8")

        print(proc.stdout, end="")
        if proc.returncode == 0:
            passes += 1
        else:
            print("(See runs/trial_{:02d} for details)".format(i))

    print(f"\nPasses: {passes}/{args.num_trials}  =>  Pass-rate: {passes/args.num_trials:.1%}")

if __name__ == "__main__":
    main()
