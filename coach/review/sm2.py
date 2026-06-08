"""SM-2 spaced repetition scheduling.

Public API:
    def sm2_schedule(quality, prev) -> dict
"""
from __future__ import annotations

# SM-2 constants (SuperMemo-2 / Anki standard)
_MIN_EF = 1.3
_DEFAULT_EF = 2.5
_PASS_THRESHOLD = 3   # quality >= 3 counts as recalled


def sm2_schedule(quality: int, prev: dict | None = None) -> dict:
    """Advance one SM-2 card state given a recall quality score.

    Args:
        quality: recall quality 0-5 (SM-2 standard scale):
                 5=perfect, 4=correct with hesitation, 3=correct but difficult,
                 2=wrong but remembered on seeing answer,
                 1=wrong, easy to remember once shown, 0=complete blackout.
        prev:    previous card state dict (None = new card). Relevant keys:
                 ef (float), interval (int days), reps (int).

    Returns dict with:
        ef        (float)  updated ease factor, clamped >= 1.3
        interval  (int)    days until next review
        reps      (int)    consecutive correct repetitions
        due_ts    (float)  Unix timestamp of next review (now + interval days)

    Algorithm (strict SM-2):
        EF' = EF + 0.1 - (5-q)*(0.08 + (5-q)*0.02)  clamped to [1.3, inf)
        quality < 3  -> reps=0, interval=1  (relearn from scratch)
        quality >= 3 -> reps += 1
                        reps==1  -> interval=1
                        reps==2  -> interval=6
                        reps>=3  -> interval=round(prev_interval * EF')
    """
    import time

    q = int(max(0, min(5, quality)))
    prev = prev or {}

    ef = float(prev.get("ef", _DEFAULT_EF))
    reps = int(prev.get("reps", 0))
    prev_interval = int(prev.get("interval", 1))

    # EF update (applied regardless of pass/fail)
    ef = ef + 0.1 - (5 - q) * (0.08 + (5 - q) * 0.02)
    ef = round(max(_MIN_EF, ef), 6)

    if q < _PASS_THRESHOLD:
        reps = 0
        interval = 1
    else:
        reps += 1
        if reps == 1:
            interval = 1
        elif reps == 2:
            interval = 6
        else:
            interval = max(1, round(prev_interval * ef))

    due_ts = time.time() + interval * 86400.0

    return {"ef": ef, "interval": interval, "reps": reps, "due_ts": due_ts}
