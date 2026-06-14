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


def _bjorklund(pulses: int, steps: int) -> list[bool]:
    """Even (Euclidean) distribution of `pulses` onsets across `steps`."""
    if steps <= 0:
        return []
    if pulses <= 0:
        return [False] * steps
    if pulses >= steps:
        return [True] * steps
    pattern_groups = [[True] for _ in range(pulses)]
    remainder_groups = [[False] for _ in range(steps - pulses)]
    while len(remainder_groups) > 1:
        count = min(len(pattern_groups), len(remainder_groups))
        for i in range(count):
            pattern_groups[i].extend(remainder_groups[i])
        if len(pattern_groups) > len(remainder_groups):
            new_remainder = pattern_groups[count:]
            pattern_groups = pattern_groups[:count]
        else:
            new_remainder = remainder_groups[count:]
        remainder_groups = new_remainder
    result: list[bool] = []
    for group in pattern_groups + remainder_groups:
        result.extend(group)
    return result


def euclidean_rhythm(pulses: int, steps: int = 16, rotation: int = 0) -> dict:
    """Build a Euclidean rhythm spreading `pulses` onsets as evenly as possible.

    Euclidean rhythms underlie countless drum and bass patterns worldwide
    (e.g. pulses=3, steps=8 is the Cuban tresillo 'O..o..o.'). The downbeat
    onset is rendered as 'O' (strong), the rest as 'o' (weak), gaps as '.';
    `rotation` shifts the whole pattern left. The pattern feeds the `rhythm`
    argument of notes_to_midi / drums_to_midi, or any drum lane.
    """
    if not isinstance(pulses, int) or isinstance(pulses, bool) or pulses < 0:
        raise ValueError(f"pulses must be a non-negative integer, got {pulses!r}")
    if not isinstance(steps, int) or isinstance(steps, bool) or not 1 <= steps <= 256:
        raise ValueError(f"steps must be an integer between 1 and 256, got {steps!r}")
    if pulses > steps:
        raise ValueError(f"pulses ({pulses}) cannot exceed steps ({steps})")
    if not isinstance(rotation, int) or isinstance(rotation, bool):
        raise ValueError(f"rotation must be an integer, got {rotation!r}")

    onsets = _bjorklund(pulses, steps)
    if rotation:
        r = rotation % steps
        onsets = onsets[r:] + onsets[:r]
    first_onset = next((i for i, on in enumerate(onsets) if on), None)
    chars = []
    for i, on in enumerate(onsets):
        if not on:
            chars.append(RHYTHM_REST)
        elif i == first_onset:
            chars.append(RHYTHM_STRONG)
        else:
            chars.append(RHYTHM_WEAK)
    pattern = "".join(chars)
    return {
        "pattern": pattern,
        "pulses": pulses,
        "steps": steps,
        "rotation": rotation,
        "legend": {"O": "strong beat", "o": "weak beat", ".": "pause"},
        "onset_positions": [i for i, on in enumerate(onsets) if on],
    }


# Named rhythm presets: (pattern, step_beats, description). 16-step patterns are
# one bar of 4/4 in sixteenth notes; 8-step patterns are one bar in eighths.
GROOVES: dict[str, tuple[str, float, str]] = {
    "four_on_floor": ("O...O...O...O...", 0.25, "A kick on every beat — house/techno/disco pulse."),
    "backbeat": ("....O.......O...", 0.25, "Snare on beats 2 and 4 — the backbone of rock and pop."),
    "offbeat": ("..o...o...o...o.", 0.25, "Hits on the off-beats (the 'and's) — house open hats, ska/reggae skank."),
    "eighths": ("o.o.o.o.o.o.o.o.", 0.25, "Steady eighth notes — driving hats or a running pulse."),
    "sixteenths": ("oooooooooooooooo", 0.25, "Steady sixteenths — rolling hats or trance arps."),
    "tresillo": ("O..o..o.", 0.25, "The 3+3+2 Cuban/Latin cell, ubiquitous in pop and reggaeton."),
    "cinquillo": ("O.oo.oo.", 0.25, "A five-stroke Cuban cell, denser sibling of the tresillo."),
    "habanera": ("O..oo.o.", 0.25, "The habanera/tango rhythm — dotted then even."),
    "son_clave_32": ("O..o..o...o.o...", 0.25, "3-2 son clave, the key pattern of Afro-Cuban son and salsa."),
    "rumba_clave_32": ("O..o...o..o.o...", 0.25, "3-2 rumba clave — like son clave but the 'three' side is shifted."),
    "bossa_nova": ("O..o..o...o..o..", 0.25, "A bossa-nova clave variant — gentle Brazilian sway."),
    "dembow": ("O..oO.o.O..oO.o.", 0.25, "The reggaeton 'dembow' — boom-ch-boom-chick engine."),
}


def groove(name: str) -> dict:
    """Return a named rhythm preset (clave, bossa, tresillo, four-on-the-floor...).

    A library of idiomatic rhythm cells as O/o/. patterns. The pattern feeds the
    `rhythm` argument of notes_to_midi / arrange tracks, or a drum lane (repeat
    it to fill more bars). Use list_grooves to see them all.
    """
    if not isinstance(name, str):
        raise ValueError(f"groove name must be a string, got {type(name).__name__}")
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    if key not in GROOVES:
        raise ValueError(f"Unknown groove {name!r}. Known grooves: {', '.join(GROOVES)}")
    pattern, step_beats, description = GROOVES[key]
    return {
        "name": key,
        "pattern": pattern,
        "step_beats": step_beats,
        "length_beats": len(pattern) * step_beats,
        "description": description,
        "legend": {"O": "strong beat", "o": "weak beat", ".": "pause"},
    }


def list_grooves() -> dict:
    """List every named rhythm preset with its pattern and description."""
    return {
        "count": len(GROOVES),
        "grooves": [
            {"name": k, "pattern": p, "step_beats": s, "description": d}
            for k, (p, s, d) in GROOVES.items()
        ],
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
