from __future__ import annotations


def latency_score_from_ms(latency_ms_p50: int) -> float:
    if latency_ms_p50 <= 0:
        return 0.5
    return max(0.1, 1.0 - (latency_ms_p50 / 10000))
