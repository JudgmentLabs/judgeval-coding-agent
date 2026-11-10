"""Hook latency for one checkout. Run it against two checkouts to compare them.

    python3 tests/bench_hook.py REPO_ROOT [RUNS] [--tracing JSON] [--against BASELINE_ROOT]

Runs REPO_ROOT/core/bin/judgment-hook RUNS times (default 200) per fixture, with
JUDGMENT_FLUSH=0 so no sender starts, and prints p50, p95, and max in ms. Uses only
the hook and the adapter maps, so it works on any checkout of this repository.
With --against, each run of REPO_ROOT alternates with one of BASELINE_ROOT, so both
see the same machine load, and the table adds both and the difference.
The fake secrets are built at runtime; the environment is synthetic.
"""

from __future__ import annotations

import json
import random
import shutil
import statistics
import string
import subprocess
import sys
import tempfile
import time
from pathlib import Path

rng = random.Random(7)


def token(n: int) -> str:
    return "".join(rng.choice(string.ascii_letters + string.digits) for _ in range(n - 1)) + "7"


def clean_output() -> str:
    """About 4 KB of ordinary build output with no secret in it: the common case."""
    lines = []
    while sum(len(line) + 1 for line in lines) < 4096:
        n = len(lines)
        lines.append(f"{n:04d} INFO build step {n} finished in {rng.randint(1, 999)} ms, max_tokens=4096 env=ci key_count={n} sha {token(40).lower()}")
    return "\n".join(lines)


def tool_output() -> str:
    secrets = [
        "sk-" + "proj-" + token(48),
        "gh" + "p_" + token(36),
        "AK" + "IA" + token(16).upper(),
        "export DEPLOY" + "_TOKEN=" + token(28),
        "postgres://app:" + token(14) + "@db:5432/app",
        "Authorization: Bearer " + token(32),
    ]
    lines = []
    while sum(len(line) + 1 for line in lines) < 4096:
        lines.append(f"{len(lines):04d} INFO build step {len(lines)} finished in {rng.randint(1, 999)} ms, max_tokens=4096 sha {token(40).lower()}")
        if len(lines) % 6 == 0:
            lines.append(secrets[(len(lines) // 6) % len(secrets)])
    return "\n".join(lines)


def fixtures(cwd: str) -> list[tuple[str, list[str], dict]]:
    claude = {"session_id": "bench-cc", "cwd": cwd, "transcript_path": "", "prompt_id": "bench-p1"}
    codex = {"session_id": "bench-cx", "turn_id": "bench-t1", "cwd": cwd, "model": "gpt-5.5-codex", "transcript_path": ""}
    cursor = {"conversation_id": "bench-cu", "generation_id": "bench-g1", "workspace_roots": [cwd], "model": "claude-sonnet-5-5"}
    output = tool_output()
    pi = {"v": 1, "harness": "pi", "session_id": "bench-pi", "cwd": cwd, "turn_id": "bench-pi:1", "environment": "production", "event": "tool",
          "tool_name": "bash", "tool_call_id": "call-1", "tool_input": {"command": "make build"}, "tool_output": output, "is_error": False,
          "start_ts": 1794000000000, "end_ts": 1794000000400, "ts": 1794000000400}
    return [
        ("claude UserPromptSubmit", ["claude-code", "UserPromptSubmit", "claude"], {**claude, "prompt": "fix the failing build and explain why it broke"}),
        ("claude PostToolUse 4KB", ["claude-code", "PostToolUse", "claude"], {**claude, "tool_name": "Bash", "tool_use_id": "tu-1",
                                                                              "tool_input": {"command": "make build"}, "tool_response": {"stdout": output, "stderr": ""}}),
        ("claude PostToolUse 4KB clean", ["claude-code", "PostToolUse", "claude"], {**claude, "tool_name": "Bash", "tool_use_id": "tu-2",
                                                                                     "tool_input": {"command": "make build"},
                                                                                     "tool_response": {"stdout": clean_output(), "stderr": ""}}),
        ("claude Stop", ["claude-code", "Stop", "claude"], {**claude, "last_assistant_message": "The build broke because a header moved. Fixed."}),
        ("codex PostToolUse", ["codex", "PostToolUse", "codex"], {**codex, "tool_name": "shell", "tool_use_id": "c-1",
                                                                 "tool_input": {"command": ["bash", "-lc", "make build"]}, "tool_response": {"output": output[:1024]}}),
        ("cursor preToolUse", ["cursor", "preToolUse", "cursor"], {**cursor, "tool_name": "Shell", "tool_use_id": "t-1", "tool_input": {"command": "make build"}}),
        ("pi --raw", [], pi),
    ]


def environment(home: Path, state: Path) -> dict:
    import os

    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(home), "LANG": "C.UTF-8", "JUDGMENT_STATE_DIR": str(state),
           "JUDGMENT_FLUSH": "0", "JUDGMENT_API_KEY": "bench" + token(32), "JUDGMENT_PROJECT_ID": "bench-project"}
    # An agent's environment is long. Most names are ordinary, a few look secret.
    for number in range(70):
        env[f"BENCH_VAR_{number}"] = f"/opt/tool/{number}/bin:{token(12)}"
    for name in ("SERVICE_TOKEN", "DB_PASSWORD", "STRIPE_SECRET_KEY", "NPM_AUTH_TOKEN"):
        env[name] = token(32)
    return env


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))]


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 2
    repo = Path(argv[0]).resolve()
    runs = int(argv[1]) if len(argv) > 1 and not argv[1].startswith("--") else 200
    tracing = {"enabled": True}
    if "--tracing" in argv:
        tracing = json.loads(argv[argv.index("--tracing") + 1])
    trees = [repo] + ([Path(argv[argv.index("--against") + 1]).resolve()] if "--against" in argv else [])
    for tree in trees:
        if not (tree / "core" / "bin" / "judgment-hook").is_file():
            print(f"no hook in {tree}", file=sys.stderr)
            return 2
    work = Path(tempfile.mkdtemp(prefix="jt-bench-"))
    try:
        home = work / "home"
        project = home / "proj"
        (project / ".judgment").mkdir(parents=True)
        (project / ".judgment" / "tracing.json").write_text(json.dumps(tracing) + "\n")
        state = work / "state"
        env = environment(home, state)
        print("  ".join(str(tree) for tree in trees) + f"  runs={runs}  tracing={json.dumps(tracing)}")
        if len(trees) == 1:
            print(f"{'fixture':<30} {'p50 ms':>8} {'p95 ms':>8} {'max ms':>8}")
        else:
            print(f"{'fixture':<30} {'base p50':>9} {'base p95':>9} {'p50':>7} {'p95':>7} {'d p50':>7} {'d p95':>7}")
        for name, spec, payload in fixtures(str(project)):
            argsets = []
            for tree in trees:
                hook = tree / "core" / "bin" / "judgment-hook"
                if spec:
                    folder, event, block = spec
                    argsets.append([str(hook), "--map", str(tree / folder / "hooks" / "map.jq"), "--event", event, "--block", block])
                else:
                    argsets.append([str(hook), "--raw"])
            data = json.dumps(payload).encode()
            for args in argsets:
                for _ in range(5):
                    subprocess.run(args, input=data, env=env, capture_output=True)
            times = [[] for _ in argsets]
            for _ in range(runs):
                for i, args in enumerate(argsets):
                    started = time.perf_counter()
                    subprocess.run(args, input=data, env=env, capture_output=True)
                    times[i].append((time.perf_counter() - started) * 1000)
                    shutil.rmtree(state / "inbox", ignore_errors=True)
            if len(trees) == 1:
                print(f"{name:<30} {statistics.median(times[0]):8.2f} {percentile(times[0], 0.95):8.2f} {max(times[0]):8.2f}")
            else:
                a50, a95 = statistics.median(times[0]), percentile(times[0], 0.95)
                b50, b95 = statistics.median(times[1]), percentile(times[1], 0.95)
                print(f"{name:<30} {b50:9.2f} {b95:9.2f} {a50:7.2f} {a95:7.2f} {a50 - b50:+7.2f} {a95 - b95:+7.2f}")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
