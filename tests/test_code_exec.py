import pandas as pd

from src.ai import code_exec


def test_run_code_assigns_result():
    df = pd.DataFrame({"a": [1, 2, 3]})
    ok, out = code_exec.run_code("result = df['a'].sum()", df)
    assert ok is True
    assert out == "6"


def test_run_code_captures_print():
    df = pd.DataFrame({"a": [1]})
    ok, out = code_exec.run_code("print(df.shape)", df)
    assert ok is True
    assert "(1, 1)" in out


def test_run_code_error_surfaces():
    df = pd.DataFrame({"a": [1]})
    ok, out = code_exec.run_code("result = 1 / 0", df)
    assert ok is False
    assert "ZeroDivisionError" in out


def test_run_code_blocks_dangerous_builtins():
    df = pd.DataFrame({"a": [1]})
    ok, out = code_exec.run_code("open('/etc/passwd')", df)
    assert ok is False
    assert "NameError" in out or "not defined" in out
