"""Sandboxed execution of LLM-generated pandas code for the RAG code interpreter.

The recalled rows are exposed to generated code as a pandas DataFrame named
``df``. Execution is restricted: only a small set of builtins is available and
no filesystem / network / subprocess access is permitted.
"""
from __future__ import annotations

import contextlib
import io

import numpy as np
import pandas as pd

# Restricted builtins available to generated code. Deliberately excludes
# open, eval, exec, compile, import, globals, locals, etc.
_SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "enumerate": enumerate, "filter": filter, "float": float, "int": int,
    "isinstance": isinstance, "len": len, "list": list, "map": map,
    "max": max, "min": min, "print": print, "range": range, "round": round,
    "set": set, "sorted": sorted, "str": str, "sum": sum, "tuple": tuple,
    "zip": zip,
}


def run_code(code: str, df: pd.DataFrame) -> tuple[bool, str]:
    """Execute *code* with ``df`` in a restricted sandbox.

    Returns ``(ok, output)`` where *output* is the captured stdout plus any
    value assigned to a ``result`` variable. On failure *ok* is ``False`` and
    *output* describes the exception.
    """
    global_ns = {"df": df, "pd": pd, "np": np, "__builtins__": _SAFE_BUILTINS}
    local_ns: dict = {}
    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            exec(code, global_ns, local_ns)  # noqa: S102 - sandboxed
        result = local_ns.get("result")
        out = stdout.getvalue().strip()
        if result is not None:
            out = (out + "\n" if out else "") + str(result)
        return True, out or "(无输出)"
    except Exception as exc:  # noqa: BLE001 - surface to caller
        return False, f"执行出错：{type(exc).__name__}: {exc}"
