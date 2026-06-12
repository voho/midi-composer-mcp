"""Seeded randomness tools: random notes from a pool, random rhythm patterns.

These are pure dice rolls — they contain no musical judgement. Pass a seed to
make a result reproducible; when no seed is given, one is drawn and reported
so any result can be regenerated.
"""

from __future__ import annotations

import random

from .notes import parse_notes

RHYTHM_STRONG = "O"
RHYTHM_WEAK = "o"
RHYTHM_REST = "."


def _make_rng(seed: int | None) -> tuple[random.Random, int]:
    if seed is None:
        seed = random.SystemRandom().randrange(2**32)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"seed must be an integer, got {seed!r}")
    return random.Random(seed), seed


def random_notes(notes: str | list, count: int = 4, allow_repeats: bool = True,
                 seed: int | None = None) -> dict:
    """Pick `count` random notes from a pool of notes.

    The pool is any list of notes — typically the `notes` output of a scale,
    chord or another tool. Octaves are kept if the pool has them.
    """
    pool = parse_notes(notes)
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError(f"count must be a positive integer, got {count!r}")
    if count > 1000:
        raise ValueError("count must be at most 1000")
    rng, seed_used = _make_rng(seed)
    if allow_repeats:
        picks = [rng.choice(pool) for _ in range(count)]
    else:
        if count > len(pool):
            raise ValueError(
                f"Cannot pick {count} distinct notes from a pool of {len(pool)}"
                f" (set allow_repeats=true or lower the count)"
            )
        picks = rng.sample(pool, count)
    return {
        "notes": [n.name for n in picks],
        "count": count,
        "pool": [n.name for n in pool],
        "allow_repeats": allow_repeats,
        "seed": seed_used,
    }


def random_rhythm(length: int = 8, density: float = 0.65,
                  accent_probability: float = 0.35, seed: int | None = None) -> dict:
    """Roll a rhythm pattern of `length` steps.

    Each step independently becomes a note with probability `density`; a note
    is strong with probability `accent_probability`. The pattern string uses
    'O' = strong beat, 'o' = weak beat, '.' = pause — the same notation the
    MIDI tools accept as a `rhythm` argument.
    """
    if not isinstance(length, int) or isinstance(length, bool) or not 1 <= length <= 256:
        raise ValueError(f"length must be an integer between 1 and 256, got {length!r}")
    for label, value in (("density", density), ("accent_probability", accent_probability)):
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value <= 1:
            raise ValueError(f"{label} must be a number between 0 and 1, got {value!r}")
    rng, seed_used = _make_rng(seed)

    steps = []
    for _ in range(length):
        if rng.random() < density:
            steps.append(RHYTHM_STRONG if rng.random() < accent_probability else RHYTHM_WEAK)
        else:
            steps.append(RHYTHM_REST)
    pattern = "".join(steps)
    return {
        "pattern": pattern,
        "length": length,
        "legend": {"O": "strong beat", "o": "weak beat", ".": "pause"},
        "note_count": sum(1 for s in steps if s != RHYTHM_REST),
        "strong_count": steps.count(RHYTHM_STRONG),
        "weak_count": steps.count(RHYTHM_WEAK),
        "rest_count": steps.count(RHYTHM_REST),
        "density": density,
        "accent_probability": accent_probability,
        "seed": seed_used,
    }


def parse_rhythm(pattern: str) -> str:
    """Validate a rhythm pattern string; whitespace is ignored."""
    if not isinstance(pattern, str):
        raise ValueError(f"rhythm must be a string of 'O', 'o' and '.', got {type(pattern).__name__}")
    cleaned = "".join(pattern.split())
    if not cleaned:
        raise ValueError("rhythm pattern is empty")
    bad = sorted(set(cleaned) - {RHYTHM_STRONG, RHYTHM_WEAK, RHYTHM_REST})
    if bad:
        raise ValueError(
            f"Invalid rhythm characters: {bad}. Use 'O' (strong beat), 'o' (weak beat), '.' (pause)."
        )
    return cleaned
