"""Check (and optionally run) the built notebooks.

  python3 tools/test-notebooks.py --syntax                 compile every code cell of every notebook
  python3 tools/test-notebooks.py --syntax week-04         one notebook
  nbvenv/bin/python tools/test-notebooks.py --run week-04  execute the cells in order, in one namespace

--run executes on this machine, not in Colab: shell/line magics (lines starting with ! or %) are skipped,
matplotlib uses the Agg backend, NB_TEST=1 is set (notebooks may use it to shorten long loops), and each
cell has a time limit (default 300 s, --timeout N). A cell fails the run if it raises. Cells whose first
line is "# nb-test: skip" are not executed (use it only for GPU-only or multi-gigabyte downloads).
"""
import argparse
import ast
import contextlib
import io
import json
import os
import re
import signal
import sys
import time
import traceback
import warnings

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NB_DIR = os.path.join(ROOT, "notebooks")


def load(name):
    with open(os.path.join(NB_DIR, name + ".ipynb"), encoding="utf-8") as f:
        nb = json.load(f)
    cells = []
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        lines = [("pass  # magic skipped: " + l.strip()) if re.match(r"^\s*[!%]", l) else l for l in src.split("\n")]
        cells.append((i, src, "\n".join(lines)))
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weeks", nargs="*")
    ap.add_argument("--syntax", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--timeout", type=int, default=300)
    a = ap.parse_args()
    if not (a.syntax or a.run):
        a.syntax = True
    names = sorted(f[:-6] for f in os.listdir(NB_DIR) if re.match(r"^week-\d\d\.ipynb$", f))
    if a.weeks:
        names = [n for n in names if n in [w.replace(".ipynb", "") for w in a.weeks]]
    failed = 0
    for name in names:
        cells = load(name)
        bad = []
        for idx, src, code in cells:
            try:
                ast.parse(code)
            except SyntaxError as e:
                bad.append(f"cell {idx}: SyntaxError line {e.lineno}: {e.msg}")
        if bad:
            failed += 1
            print(f"✗ {name} syntax")
            [print("    " + b) for b in bad]
            continue
        if not a.run:
            print(f"✓ {name}  {len(cells)} code cells compile")
            continue
        import matplotlib
        matplotlib.use("Agg")
        warnings.filterwarnings("ignore")
        os.environ["NB_TEST"] = "1"
        ns = {"__name__": "__main__"}
        def alarm(*_):
            raise TimeoutError(f"cell took longer than {a.timeout}s")
        signal.signal(signal.SIGALRM, alarm)
        t0 = time.time()
        ran = skipped = 0
        err = None
        cwd = os.getcwd()
        os.chdir("/tmp")
        for idx, src, code in cells:
            if src.lstrip().startswith("# nb-test: skip"):
                skipped += 1
                continue
            try:
                signal.alarm(a.timeout)
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(compile(code, f"{name}[cell {idx}]", "exec"), ns)
                signal.alarm(0)
                ran += 1
            except BaseException as e:  # noqa: BLE001
                signal.alarm(0)
                tb = [t for t in traceback.extract_tb(e.__traceback__) if t.filename.startswith(name)]
                line = tb[-1].lineno if tb else "?"
                snippet = code.split("\n")[line - 1].strip()[:90] if isinstance(line, int) and line <= len(code.split("\n")) else ""
                err = f"cell {idx} line {line}: {type(e).__name__}: {str(e)[:200]}   >> {snippet}"
                break
        os.chdir(cwd)
        if err:
            failed += 1
            print(f"✗ {name} RUN  ({ran} cells ok, then failed) {err}")
        else:
            print(f"✓ {name} RAN  {ran} cells in {time.time() - t0:.0f}s" + (f" ({skipped} skipped)" if skipped else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
