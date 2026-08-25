"""
Per-agent latency tracking (observability).

Every node in the graph gets wrapped with @track_latency("agent_name") so we
know exactly how much time each of the six agents spends, instead of only a
single end-to-end number. Used for:
  - the Fig-style "per-agent latency breakdown" table in the paper
  - spotting the actual bottleneck (LLM calls vs retrieval vs I/O)

Safe for parallel LangGraph branches: each node writes to its OWN key in
agent_timings, so there is no read/write conflict when nodes run in the same
superstep.
"""
import time
import functools


def track_latency(agent_name: str):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(state, *args, **kwargs):
            start = time.perf_counter()
            result = fn(state, *args, **kwargs)
            elapsed = time.perf_counter() - start

            timings = dict(state.get("agent_timings") or {})
            timings[agent_name] = round(elapsed, 4)

            if isinstance(result, dict):
                result = {**result, "agent_timings": timings}
            return result
        return wrapper
    return decorator
