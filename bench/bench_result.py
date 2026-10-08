"""Micro-benchmarks for the Result Monad (run: `uv run --no-sync python bench/bench_result.py`).

Covers a 10-step map/bind chain on Ok and on Err, `==` on large payloads, and the per-instance
size. Pass pyperf options such as `--fast` or `-o out.json` to compare runs with
`python -m pyperf compare_to before.json after.json`.
"""

import sys

import pyperf

from pymoliath.result import Err, Ok, Result


def inc(x: int) -> int:
    return x + 1


def step(x: int) -> Result[int, str]:
    return Ok(x + 1)


def chain(value: Result[int, str]) -> Result[int, str]:
    return (
        value.map(inc)
        .bind(step)
        .map(inc)
        .bind(step)
        .map(inc)
        .bind(step)
        .map(inc)
        .bind(step)
        .map(inc)
        .bind(step)
    )


def main() -> None:
    ok: Result[int, str] = Ok(1)
    err: Result[int, str] = Err("error")
    payload = list(range(10_000))
    left, right = Ok(payload), Ok(list(payload))

    runner = pyperf.Runner()
    runner.metadata["sizeof_ok"] = str(sys.getsizeof(ok))
    runner.metadata["sizeof_err"] = str(sys.getsizeof(err))
    runner.bench_func("chain-10-ok", chain, ok)
    runner.bench_func("chain-10-err", chain, err)
    runner.bench_func("eq-large-payload", left.__eq__, right)


if __name__ == "__main__":
    main()
